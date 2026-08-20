# Enterprise Platform Operational Limits & Architecture Specifications

**Effective Date:** February 1, 2025  
**Document ID:** ENG-LIM-2025  
**Department:** Engineering / Architecture  

## 1. API Rate Limits
- **Standard Tier:** 1,000 requests per minute per IP address.
- **Enterprise Tier:** 50,000 requests per minute per tenant namespace.

## 2. Document Upload & Storage Limits
- Maximum single file upload size: **100 MB** for Standard tier, **2 GB** for Enterprise tier.
- Total tenant storage limit: **500 GB** for Standard, **50 TB** for Enterprise tier.

## 3. Database Query & Concurrency Limits
- Maximum concurrent database read queries: 250 connections.
- Query execution timeout limit: 30 seconds per single transaction.

## 4. User Account & Seat Limits
- Maximum seats per Standard organization: 50 active users.
- Enterprise plans support unlimited seat provisions with SAML 2.0 Single Sign-On (SSO).
