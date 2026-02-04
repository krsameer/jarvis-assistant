"""
Example usage script for Jarvis AI Assistant
Demonstrates how to use the API programmatically.
"""

import requests
import json
import time

# API Configuration
API_BASE_URL = "http://localhost:8000"


def check_health():
    """Check if the API is healthy."""
    print("🏥 Checking system health...")
    response = requests.get(f"{API_BASE_URL}/health")
    health = response.json()
    
    print(f"  Status: {health['status']}")
    print(f"  Services:")
    for service, status in health['services'].items():
        icon = "✅" if status else "❌"
        print(f"    {icon} {service}")
    print()


def ingest_sample_documents():
    """Ingest sample enterprise documents."""
    print("📄 Ingesting sample documents...\n")
    
    documents = [
        {
            "text": """
            Employee Vacation Policy
            
            All full-time employees are entitled to paid vacation time based on tenure:
            - 0-2 years of service: 15 days per year
            - 3-5 years of service: 20 days per year  
            - 6+ years of service: 25 days per year
            
            Vacation requests must be submitted at least 2 weeks in advance through the HR portal.
            Unused vacation days can be carried over, up to a maximum of 5 days per year.
            Vacation time accrues monthly at a rate of 1/12th of the annual allowance.
            
            In case of emergency, vacation can be requested with shorter notice, subject to manager approval.
            """,
            "metadata": {
                "source": "employee_handbook.pdf",
                "section": "benefits",
                "type": "policy",
                "page": 12
            }
        },
        {
            "text": """
            Remote Work Policy
            
            Our company supports flexible work arrangements. Employees may work remotely based on the following guidelines:
            
            Eligibility:
            - Must have completed 3 months probation period
            - Role must be suitable for remote work
            - Must have reliable internet connection (minimum 10 Mbps)
            
            Requirements:
            - Must be available during core hours (10am-3pm local time)
            - Respond to messages within 2 hours during work hours
            - Attend all team meetings via video conference
            - Maintain secure home office setup with VPN access
            
            Frequency:
            - Hybrid roles: 2-3 days remote per week
            - Fully remote roles: 100% remote with quarterly in-office meetings
            
            Equipment:
            - Company provides laptop, monitor, and necessary peripherals
            - Employees responsible for internet and workspace
            """,
            "metadata": {
                "source": "employee_handbook.pdf",
                "section": "work_arrangements",
                "type": "policy",
                "page": 18
            }
        },
        {
            "text": """
            SaaS Platform - API Authentication
            
            Our platform API uses OAuth 2.0 for authentication. Here's how to get started:
            
            1. Register your application:
               - Log into the developer portal at https://developer.company.com
               - Create a new application
               - Note your Client ID and Client Secret
            
            2. Request an access token:
               POST https://api.company.com/oauth/token
               Content-Type: application/x-www-form-urlencoded
               
               client_id=YOUR_CLIENT_ID
               client_secret=YOUR_CLIENT_SECRET
               grant_type=client_credentials
            
            3. Use the access token:
               Authorization: Bearer YOUR_ACCESS_TOKEN
            
            Token Properties:
            - Tokens expire after 1 hour
            - Refresh tokens are valid for 30 days
            - Rate limit: 1000 requests per hour per token
            
            Security Best Practices:
            - Never expose client secrets in frontend code
            - Use HTTPS for all API requests
            - Rotate credentials every 90 days
            - Implement token refresh logic
            """,
            "metadata": {
                "source": "api_documentation.pdf",
                "section": "authentication",
                "type": "technical_doc",
                "page": 5
            }
        },
        {
            "text": """
            Expense Reimbursement Policy
            
            Employees can submit expenses for reimbursement through our expense management system.
            
            Eligible Expenses:
            - Travel costs (flights, hotels, ground transportation)
            - Client meals and entertainment
            - Office supplies and equipment
            - Professional development (courses, certifications, books)
            - Home office setup (up to $500 per year)
            
            Submission Process:
            1. Take photos/save receipts for all expenses
            2. Log into expense portal at https://expenses.company.com
            3. Create new expense report
            4. Upload receipts and categorize expenses
            5. Submit for manager approval
            
            Approval Timeline:
            - Manager review: 3 business days
            - Finance approval: 2 business days
            - Reimbursement: 5 business days after final approval
            
            Limits:
            - Meals: $75 per person
            - Hotel: $250 per night
            - No pre-approval needed for expenses under $100
            - Expenses over $500 require director approval
            
            All expenses must be submitted within 30 days of purchase.
            """,
            "metadata": {
                "source": "finance_policies.pdf",
                "section": "expenses",
                "type": "policy",
                "page": 8
            }
        }
    ]
    
    # Ingest each document
    for i, doc in enumerate(documents, 1):
        print(f"  Document {i}/{len(documents)}: {doc['metadata']['source']}")
        
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json=doc,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"    ✅ Ingested {result['chunks_processed']} chunks")
        else:
            print(f"    ❌ Failed: {response.text}")
        
        time.sleep(1)  # Be nice to the API
    
    print()


def ask_questions():
    """Ask sample questions to demonstrate RAG."""
    print("💬 Asking questions...\n")
    
    questions = [
        "How many vacation days do employees with 4 years of service get?",
        "What are the requirements for working remotely?",
        "How do I authenticate with the API?",
        "What's the limit for meal expenses?",
        "Can I carry over unused vacation days?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"query": question},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result['response']
            sources = result.get('sources', [])
            
            print(f"A{i}: {answer}\n")
            
            if sources:
                print(f"    📚 Sources ({len(sources)}):")
                for j, source in enumerate(sources[:2], 1):  # Show top 2 sources
                    metadata = source.get('metadata', {})
                    print(f"      {j}. {metadata.get('source', 'Unknown')} "
                          f"(Score: {source['score']:.3f})")
                print()
        else:
            print(f"❌ Error: {response.text}\n")
        
        time.sleep(2)  # Be nice to the API


def get_stats():
    """Get system statistics."""
    print("📊 System Statistics:")
    
    response = requests.get(f"{API_BASE_URL}/stats")
    
    if response.status_code == 200:
        stats = response.json()
        
        print(f"\n  Embedding Model:")
        print(f"    • Model: {stats['embedding_model']['model_name']}")
        print(f"    • Dimension: {stats['embedding_model']['dimension']}")
        
        print(f"\n  LLM Model:")
        print(f"    • Model: {stats['llm_model']['model']}")
        print(f"    • Temperature: {stats['llm_model']['temperature']}")
        
        print(f"\n  Vector Store:")
        print(f"    • Total Vectors: {stats['vector_store']['total_vector_count']}")
        print(f"    • Dimension: {stats['vector_store']['dimension']}")
        
        print()
    else:
        print(f"  ❌ Could not retrieve stats\n")


def main():
    """Main execution flow."""
    print("=" * 70)
    print("🤖 Jarvis AI Assistant - Example Usage")
    print("=" * 70)
    print()
    
    try:
        # Check system health
        check_health()
        
        # Ingest sample documents
        ingest_sample_documents()
        
        # Wait for indexing
        print("⏳ Waiting for indexing to complete...")
        time.sleep(3)
        print()
        
        # Ask questions
        ask_questions()
        
        # Get system stats
        get_stats()
        
        print("=" * 70)
        print("✅ Demo complete!")
        print("=" * 70)
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API")
        print("   Make sure the backend is running:")
        print("   cd backend && uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
