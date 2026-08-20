# Enterprise Security, Data Access Control & RBAC Policy

**Effective Date:** March 1, 2025  
**Document ID:** SEC-RBAC-2025  
**Department:** Human Resources, Finance, Legal, Engineering  
**Classification:** Restricted / Confidential  

## 1. Access Control Matrix & Departmental Scoping
Document visibility and knowledge base retrieval are strictly enforced via Microsoft Entra ID claims and Role-Based Access Control (RBAC):
- **HR Department:** Access to employee compensation, performance reviews, personal health records, and unredacted payroll documents.
- **Finance Department:** Access to fiscal audits, revenue projections, tax filings, and banking wire details.
- **Engineering Department:** Access to technical specifications, architecture blueprints, API documentation, and infrastructure credentials.
- **Legal Department:** Access to litigation records, contracts, regulatory compliance filings, and NDA archives.

## 2. Cross-Departmental Isolation Security Rule
Under strict security compliance (SOC2 Type II & ISO 27001), users belonging to Engineering **MUST NEVER** be granted access to HR personnel files, financial audit logs, or confidential legal litigation documents.

## 3. Data Encryption Standards
All knowledge base indexes and vectors are encrypted at rest using Customer-Managed Keys (CMK) via Azure Key Vault with 256-bit AES encryption.
