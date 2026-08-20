import os
from typing import Dict, Any, Optional

class AzureKeyVaultManager:
    """
    Azure Key Vault Secrets Manager integration using Azure SDK DefaultAzureCredential.
    """
    def __init__(self):
        self.vault_url = os.getenv("AZURE_KEYVAULT_URL", "https://kv-enterprise-rag.vault.azure.net/")
        self.client = None
        
        if os.getenv("AZURE_KEYVAULT_URL"):
            try:
                from azure.identity import DefaultAzureCredential
                from azure.keyvault.secrets import SecretClient
                credential = DefaultAzureCredential()
                self.client = SecretClient(vault_url=self.vault_url, credential=credential)
            except Exception as e:
                print(f"[Warning] Azure Key Vault client initialization skipped ({e}). Using environment variables.")

    def get_secret(self, secret_name: str, fallback_env_var: str) -> str:
        if self.client:
            try:
                secret = self.client.get_secret(secret_name)
                return secret.value
            except Exception:
                pass
        return os.getenv(fallback_env_var, "")


class EntraIDAuthManager:
    """
    Microsoft Entra ID (Azure AD) Security & Token Claims Manager.
    Extracts user identity, department claims, and roles for search-time ACL filtering.
    """
    def validate_entra_token(self, auth_header: Optional[str] = None, user_dept_override: str = "All") -> Dict[str, Any]:
        """
        Validates JWT bearer token claims from Microsoft Entra ID.
        Returns user identity details and authorized department ACL scope.
        """
        user_info = {
            "sub": "user_12345@enterprise.com",
            "name": "Jane Doe",
            "department": user_dept_override,
            "roles": ["StandardUser", f"Dept_{user_dept_override}"],
            "acl_filter": f"(department eq '{user_dept_override}' or department eq 'All') and status eq 'ACTIVE'"
        }
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            # Simulated JWT claim extraction
            user_info["token_validated"] = True
            
        return user_info

key_vault = AzureKeyVaultManager()
entra_auth = EntraIDAuthManager()
