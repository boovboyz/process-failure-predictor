"""
Hyperparameter tuning with Optuna for process prediction models.
Provides automatic optimization of XGBoost and LightGBM parameters.
"""
import numpy as np
from typing import Dict, Any, Optional, Literal, Callable
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class HyperparameterTuner:
    """
    Optuna-based hyperparameter tuner for process prediction models.
    
    Supports tuning for:
    - XGBoost classifiers and regressors
    - LightGBM classifiers and regressors
    
    Uses Bayesian optimization (TPE) for efficient search.
    """
    
    def __init__(
        self,
        model_type: Literal["xgboost", "lightgbm"] = "xgboost",
        task_type: Literal["classification", "regression"] = "classification",
        n_trials: int = 50,
        timeout: Optional[int] = None,
        metric: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize the tuner.
        
        Args:
            model_type: Model backend ("xgboost" or "lightgbm")
            task_type: Task type ("classification" or "regression")
            n_trials: Number of optimization trials
            timeout: Optional timeout in seconds
            metric: Metric to optimize (default: accuracy for classification, mae for regression)
            random_state: Random seed for reproducibility
            verbose: Whether to show optimization progress
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna not installed. Run: pip install optuna")
        
        self.model_type = model_type
        self.task_type = task_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.verbose = verbose
        
        # Set default metric
        if metric is None:
            self.metric = "accuracy" if task_type == "classification" else "mae"
        else:
            self.metric = metric
        
        # Best parameters found
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = 0.0
        self.study: Optional[optuna.Study] = None
    
    def _get_xgboost_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get XGBoost hyperparameter search space."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': self.random_state,
            'verbosity': 0,
        }
    
    def _get_lightgbm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get LightGBM hyperparameter search space."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': self.random_state,
            'verbose': -1,
            'force_col_wise': True,
        }
    
    def _create_model(self, params: Dict[str, Any], n_classes: int = 2):
        """Create a model with the given parameters."""
        if self.model_type == "xgboost":
            if self.task_type == "classification":
                if n_classes == 2:
                    params['objective'] = 'binary:logistic'
                    params['eval_metric'] = 'logloss'
                else:
                    params['objective'] = 'multi:softprob'
                    params['num_class'] = n_classes
                    params['eval_metric'] = 'mlogloss'
                return xgb.XGBClassifier(**params)
            else:
                params['objective'] = 'reg:squarederror'
                params['eval_metric'] = 'mae'
                return xgb.XGBRegressor(**params)
        else:  # lightgbm
            if self.task_type == "classification":
                if n_classes == 2:
                    params['objective'] = 'binary'
                    params['metric'] = 'binary_logloss'
                else:
                    params['objective'] = 'multiclass'
                    params['num_class'] = n_classes
                    params['metric'] = 'multi_logloss'
                return lgb.LGBMClassifier(**params)
            else:
                params['objective'] = 'regression'
                params['metric'] = 'mae'
                return lgb.LGBMRegressor(**params)
    
    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluate predictions using the configured metric."""
        if self.task_type == "classification":
            if self.metric == "accuracy":
                return accuracy_score(y_true, y_pred)
            elif self.metric == "f1":
                return f1_score(y_true, y_pred, average='weighted')
            else:
                return accuracy_score(y_true, y_pred)
        else:
            if self.metric == "mae":
                return -mean_absolute_error(y_true, y_pred)  # Negative for maximization
            else:
                return -mean_absolute_error(y_true, y_pred)
    
    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int = 2,
        val_ratio: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target labels
            n_classes: Number of classes (for classification)
            val_ratio: Validation split ratio
            
        Returns:
            Dictionary with best parameters
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=self.random_state
        )
        
        def objective(trial: optuna.Trial) -> float:
            # Get parameters based on model type
            if self.model_type == "xgboost":
                params = self._get_xgboost_params(trial)
            else:
                params = self._get_lightgbm_params(trial)
            
            # Create and train model
            model = self._create_model(params, n_classes)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            score = self._evaluate(y_val, y_pred)
            
            return score
        
        # Create study
        direction = "maximize"  # We negate MAE for regression
        sampler = TPESampler(seed=self.random_state)
        
        if self.verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler,
        )
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=self.verbose,
        )
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params
    
    def get_best_model(self, n_classes: int = 2):
        """
        Create a model with the best found parameters.
        
        Args:
            n_classes: Number of classes (for classification)
            
        Returns:
            Configured model with best parameters
        """
        if not self.best_params:
            raise ValueError("No tuning has been performed yet. Call tune() first.")
        
        params = {**self.best_params}
        params['random_state'] = self.random_state
        
        if self.model_type == "xgboost":
            params['verbosity'] = 0
        else:
            params['verbose'] = -1
            params['force_col_wise'] = True
        
        return self._create_model(params, n_classes)
    
    def get_tuning_summary(self) -> Dict[str, Any]:
        """Get a summary of the tuning process."""
        if self.study is None:
            return {"status": "not_run"}
        
        return {
            "status": "completed",
            "n_trials": len(self.study.trials),
            "best_score": self.best_score,
            "best_params": self.best_params,
            "optimization_history": [t.value for t in self.study.trials if t.value is not None],
        }


def quick_tune(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "xgboost",
    task_type: str = "classification",
    n_classes: int = 2,
    n_trials: int = 30,
) -> Dict[str, Any]:
    """
    Quick hyperparameter tuning with sensible defaults.
    
    Args:
        X: Feature matrix
        y: Target labels
        model_type: "xgboost" or "lightgbm"
        task_type: "classification" or "regression"
        n_classes: Number of classes
        n_trials: Number of optimization trials
        
    Returns:
        Best parameters found
    """
    tuner = HyperparameterTuner(
        model_type=model_type,
        task_type=task_type,
        n_trials=n_trials,
        verbose=False,
    )
    
    return tuner.tune(X, y, n_classes=n_classes)
