#!/bin/bash
# Deploy Process Failure Predictor to EC2
# Run this script on a fresh Ubuntu 22.04 EC2 instance

set -e

echo "🚀 Deploying Process Failure Predictor..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Install Docker Compose
echo "🐳 Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo apt-get install -y docker-compose-plugin
fi

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp backend/.env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file to add your ANTHROPIC_API_KEY (optional)"
    echo "   Run: nano .env"
    echo ""
fi

# Build and start containers
echo "🏗️  Building and starting containers..."
sudo docker compose up -d --build

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check status
echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Container Status:"
sudo docker compose ps
echo ""
echo "🌐 Access your application at: http://$(curl -s ifconfig.me)"
echo ""
echo "📝 Useful commands:"
echo "   View logs:     sudo docker compose logs -f"
echo "   Stop:          sudo docker compose down"
echo "   Restart:       sudo docker compose restart"
