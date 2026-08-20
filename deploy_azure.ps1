# Azure Resource Deployment & App Service Provisioning Script
# Note: This script provisions the Azure App Service Web App hosting layer.
# To provision the Azure AI Search index schema, execute: python setup_azure_search.py

param (
    [string]$ResourceGroupName = "rg-enterprise-rag-prod",
    [string]$Location = "eastus",
    [string]$AppName = "enterprise-rag-assistant-$((Get-Random))",
    [string]$AppServicePlan = "asp-enterprise-rag-prod"
)

Write-Host "🚀 Creating Azure Resource Group '$ResourceGroupName' in '$Location'..."
az group create --name $ResourceGroupName --location $Location

Write-Host "⚡ Creating Azure App Service Plan (B1 Linux)..."
az appservice plan create --name $AppServicePlan --resource-group $ResourceGroupName --sku B1 --is-linux

Write-Host "🌐 Provisioning Azure Web App '$AppName'..."
az webapp create --resource-group $ResourceGroupName --plan $AppServicePlan --name $AppName --runtime "PYTHON:3.11"

Write-Host "⚙️ Setting Streamlit Startup Command..."
az webapp config set --resource-group $ResourceGroupName --name $AppName --startup-file "python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0"

Write-Host "✅ Azure App Service Web App Provisioned! URL: https://$AppName.azurewebsites.net"
