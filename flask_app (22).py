# app.py
from flask import Flask, render_template_string, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import json
import os
import uuid
import requests
from functools import wraps
from datetime import datetime
import csv
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# JSON database files
USERS_FILE = 'users.json'
AI_WORKFORCE_FILE = 'ai_workforce.json'
API_KEYS_FILE = 'api_keys.json'
TELEGRAM_BOTS_FILE = 'telegram_bots.json'
WHATSAPP_BOTS_FILE = 'whatsapp_bots.json'
FACEBOOK_BOTS_FILE = 'facebook_tokens.json'
CHAT_HISTORY_FILE = 'chat_history.json'
USER_DATABASES_FILE = 'user_databases.json'
# Add to your existing JSON database files
SCHEDULED_POSTS_FILE = 'scheduled_posts.json'
CONTENT_STRATEGY_FILE = 'content_strategy.json'
AUTO_REPLY_RULES_FILE = 'auto_reply_rules.json'

# Configuration
UPLOAD_FOLDER = 'user_databases'
ALLOWED_EXTENSIONS = {'json', 'csv'}
FACEBOOK_VERIFY_TOKEN = os.environ.get('FACEBOOK_VERIFY_TOKEN', 'your-verify-token-here')
FACEBOOK_APP_ID = os.environ.get('FACEBOOK_APP_ID', 'YOUR_FACEBOOK_APP_ID')
FACEBOOK_APP_SECRET = os.environ.get('FACEBOOK_APP_SECRET', 'YOUR_FACEBOOK_APP_SECRET')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Available AI services
AVAILABLE_AI_SERVICES = [
    {"id": "whatsapp", "name": "WhatsApp Customer Care", "description": "AI-powered customer support for WhatsApp"},
    {"id": "telegram", "name": "Telegram Assistant", "description": "AI assistant for Telegram messaging"},
    {"id": "facebook", "name": "Facebook Manager", "description": "AI for Facebook content and engagement"},
    {"id": "instagram", "name": "Instagram Manager", "description": "AI for Instagram content and engagement"},
    {"id": "twitter", "name": "Twitter Manager", "description": "AI for Twitter/X content and engagement"},
    {"id": "security", "name": "Chief Security AI Agent", "description": "AI security monitoring and threat detection"},
    {"id": "project_management", "name": "Project Management AI", "description": "AI-powered project planning and tracking"},
    {"id": "sales", "name": "Sales Assistant AI", "description": "AI for sales support and customer engagement"},
    {"id": "hr", "name": "HR Assistant AI", "description": "AI for human resources and employee support"}
]

# Initialize JSON files if they don't exist
def init_json_files():
    files = [
        USERS_FILE, AI_WORKFORCE_FILE, API_KEYS_FILE, TELEGRAM_BOTS_FILE,
        WHATSAPP_BOTS_FILE, FACEBOOK_BOTS_FILE, CHAT_HISTORY_FILE,
        USER_DATABASES_FILE, SCHEDULED_POSTS_FILE, CONTENT_STRATEGY_FILE,
        AUTO_REPLY_RULES_FILE  # Add this new file
    ]

    for file in files:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                json.dump({}, f)

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.chmod(UPLOAD_FOLDER, 0o755)

# Load data from JSON files
def load_data(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Save data to JSON files
def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Check if user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Get user's API key for a specific service
def get_user_api_key(user_id, service_name):
    api_keys = load_data(API_KEYS_FILE)
    for key_id, key_data in api_keys.items():
        if key_data['user_id'] == user_id and key_data['service_name'] == service_name:
            return key_data['api_key']
    return None

# Save chat history
def save_chat_history(user_id, ai_id, platform, sender_id, message, response, direction="incoming"):
    chat_history = load_data(CHAT_HISTORY_FILE)

    # Create a unique conversation ID
    conversation_id = f"{user_id}_{ai_id}_{platform}_{sender_id}"

    if conversation_id not in chat_history:
        chat_history[conversation_id] = {
            'user_id': user_id,
            'ai_id': ai_id,
            'platform': platform,
            'sender_id': sender_id,
            'messages': [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

    chat_history[conversation_id]['messages'].append({
        'direction': direction,
        'message': message,
        'response': response,
        'timestamp': datetime.now().isoformat()
    })

    # Keep only the last 100 messages per conversation
    if len(chat_history[conversation_id]['messages']) > 100:
        chat_history[conversation_id]['messages'] = chat_history[conversation_id]['messages'][-100:]

    chat_history[conversation_id]['updated_at'] = datetime.now().isoformat()

    save_data(chat_history, CHAT_HISTORY_FILE)

# Get chat history for a user
def get_user_chat_history(user_id, ai_id=None, platform=None):
    chat_history = load_data(CHAT_HISTORY_FILE)
    user_chats = {}

    for conv_id, conv_data in chat_history.items():
        if conv_data['user_id'] == user_id:
            if ai_id is None or conv_data['ai_id'] == ai_id:
                if platform is None or conv_data['platform'] == platform:
                    user_chats[conv_id] = conv_data

    return user_chats

def analyze_database_structure(data):
    """Enhanced analysis that handles wide database structures"""
    if isinstance(data, dict):
        # This is a wide database structure (like your template)
        structure_analysis = {}

        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                # It's a table/array - analyze first item
                if isinstance(value[0], dict):
                    structure_analysis[key] = {
                        'type': 'table',
                        'record_count': len(value),
                        'fields': analyze_record_structure(value[0])
                    }
                else:
                    structure_analysis[key] = {
                        'type': 'array',
                        'record_count': len(value),
                        'sample_value': str(value[0])[:100]
                    }
            elif isinstance(value, dict):
                # It's a nested object
                structure_analysis[key] = {
                    'type': 'object',
                    'fields': list(value.keys())
                }
            else:
                # It's a simple field
                structure_analysis[key] = {
                    'type': 'field',
                    'value_type': type(value).__name__,
                    'sample_value': str(value)[:100]
                }

        return structure_analysis

    elif isinstance(data, list) and len(data) > 0:
        # Traditional flat database
        first_record = data[0]
        if isinstance(first_record, dict):
            return analyze_record_structure(first_record)

    return {}

def analyze_record_structure(record):
    """Analyze structure of a single record"""
    field_analysis = {}
    for key, value in record.items():
        field_analysis[key] = {
            'type': type(value).__name__,
            'sample_value': str(value)[:50] + '...' if len(str(value)) > 50 else str(value)
        }
    return field_analysis

# Add these functions after your existing utility functions

def query_user_database(user_id, query_params):
    """n8n-inspired query system for JSON/CSV data"""
    user_databases = load_data(USER_DATABASES_FILE)
    results = []

    for db_id, db_data in user_databases.items():
        if db_data['user_id'] == user_id and db_data.get('data'):
            data = db_data['data']

            # Apply filters like n8n's filter node
            filtered_data = apply_filters(data, query_params.get('filters', {}))

            # Apply sorting
            sorted_data = apply_sorting(filtered_data, query_params.get('sort', {}))

            # Apply pagination
            paginated_data = apply_pagination(sorted_data, query_params.get('pagination', {}))

            results.append({
                'database': db_data['name'],
                'data': paginated_data,
                'total_count': len(filtered_data)
            })

    return results

def apply_filters(data, filters):
    """n8n-style filtering"""
    if not filters or not isinstance(data, list):
        return data

    filtered_data = []
    for item in data:
        if isinstance(item, dict):
            match = True
            for field, condition in filters.items():
                if field in item:
                    if not evaluate_condition(item[field], condition):
                        match = False
                        break
            if match:
                filtered_data.append(item)

    return filtered_data

def evaluate_condition(value, condition):
    """Evaluate n8n-style conditions"""
    if isinstance(condition, dict):
        # Handle complex conditions: {operation: value}
        for op, op_value in condition.items():
            if op == 'equals':
                return str(value) == str(op_value)
            elif op == 'contains':
                return str(op_value).lower() in str(value).lower()
            elif op == 'greaterThan':
                try:
                    return float(value) > float(op_value)
                except:
                    return False
            elif op == 'lessThan':
                try:
                    return float(value) < float(op_value)
                except:
                    return False
            # Add more operations as needed
    else:
        # Simple equality check
        return str(value) == str(condition)

    return False

def apply_sorting(data, sort_config):
    """n8n-style sorting"""
    if not sort_config or not isinstance(data, list) or len(data) == 0:
        return data

    field = sort_config.get('field')
    direction = sort_config.get('direction', 'asc')

    if not field or field not in data[0]:
        return data

    try:
        return sorted(data, key=lambda x: str(x.get(field, '')), reverse=(direction == 'desc'))
    except:
        return data

def apply_pagination(data, pagination_config):
    """n8n-style pagination"""
    if not isinstance(data, list):
        return data

    page = pagination_config.get('page', 1)
    limit = pagination_config.get('limit', 10)

    start = (page - 1) * limit
    end = start + limit

    return data[start:end]

def analyze_prompt_for_data(prompt):
    """n8n-style prompt analysis to detect data needs"""
    prompt_lower = prompt.lower()
    requirements = {
        'needs_query': False,
        'likely_tables': [],
        'expected_fields': []
    }

    # Detect database mentions (like n8n's pattern matching)
    database_keywords = ['customer', 'product', 'order', 'sales', 'inventory', 'user', 'client', 'item']
    for keyword in database_keywords:
        if keyword in prompt_lower:
            requirements['likely_tables'].append(keyword)
            requirements['needs_query'] = True

    # Detect field mentions
    field_patterns = [
        ('name', ['name', 'contact', 'person']),
        ('email', ['email', 'address']),
        ('price', ['price', 'cost', 'amount']),
        ('date', ['date', 'time', 'when']),
        ('quantity', ['quantity', 'number', 'count']),
        ('status', ['status', 'state'])
    ]

    for field_name, patterns in field_patterns:
        if any(pattern in prompt_lower for pattern in patterns):
            requirements['expected_fields'].append(field_name)

    return requirements

def automatically_execute_query(available_databases, requirements, prompt):
    """Execute queries that work with both wide and flat databases"""
    results = {}

    for db in available_database:
        db_data = db.get('data', {})
        db_name = db['name']

        # Handle wide database structure
        if isinstance(db_data, dict) and db.get('structure_type') == 'wide':
            wide_results = query_wide_database(db_data, prompt, requirements)
            if wide_results:
                results[db_name] = wide_results

        # Handle flat database structure
        elif isinstance(db_data, list):
            flat_results = query_flat_database(db_data, prompt, requirements)
            if flat_results:
                results[db_name] = flat_results

        # Handle single object database
        elif isinstance(db_data, dict):
            # Check if this single object matches the query
            if does_object_match_query(db_data, prompt, requirements):
                results[db_name] = [db_data]

    return results

def query_wide_database(wide_data, prompt, requirements):
    """Query databases with wide structure (multiple tables in one object)"""
    prompt_lower = prompt.lower()
    results = []

    # Determine which table(s) to query based on prompt
    target_tables = identify_target_tables(wide_data, prompt_lower)

    for table_name in target_tables:
        table_data = wide_data.get(table_name)

        if isinstance(table_data, list):
            # Query this table
            table_results = query_table(table_data, prompt, requirements, table_name)
            if table_results:
                results.extend(table_results)

        elif isinstance(table_data, dict):
            # Single record table
            if does_object_match_query(table_data, prompt, requirements):
                results.append(table_data)

    return results

def identify_target_tables(wide_data, prompt_lower):
    """Identify which tables in wide database to query"""
    table_mappings = {
        'customer': ['customers', 'clients', 'users'],
        'product': ['products', 'items', 'inventory', 'services'],
        'order': ['orders', 'purchases', 'transactions'],
        'employee': ['employees', 'staff', 'team'],
        'store': ['store_info', 'locations', 'branches'],
        'business': ['business_info', 'company_info']
    }

    target_tables = []

    # Check prompt for table keywords
    for table_type, table_names in table_mappings.items():
        if any(keyword in prompt_lower for keyword in [table_type] + table_names):
            # Find matching table in database
            for table_name in table_names:
                if table_name in wide_data:
                    target_tables.append(table_name)
                    break

    # If no specific table found, try all tables
    if not target_tables:
        for key, value in wide_data.items():
            if isinstance(value, (list, dict)):
                target_tables.append(key)

    return target_tables

def query_table(table_data, prompt, requirements, table_name):
    """Query a specific table"""
    if not isinstance(table_data, list):
        return []

    filtered_data = table_data.copy()

    # Apply intelligent filters
    filtered_data = apply_intelligent_filters(filtered_data, prompt, requirements)

    # If too many results, limit to most relevant
    if len(filtered_data) > 20:
        filtered_data = filtered_data[:10]  # Show top 10 most relevant

    return filtered_data

def does_object_match_query(obj, prompt, requirements):
    """Check if a single object matches the query"""
    if not isinstance(obj, dict):
        return False

    prompt_lower = prompt.lower()

    # Check if object contains any of the expected fields
    if requirements['expected_fields']:
        has_expected_fields = any(field in obj for field in requirements['expected_fields'])
        if not has_expected_fields:
            return False

    # Check if object values match prompt keywords
    for value in obj.values():
        if isinstance(value, str) and any(keyword in value.lower() for keyword in requirements['likely_tables']):
            return True

    return False

def query_nested_database(nested_data, prompt, requirements):
    """Query databases with nested structure like your shoe shop example"""
    prompt_lower = prompt.lower()
    results = []

    # Check what type of data is being requested
    if any(keyword in prompt_lower for keyword in ['customer', 'client', 'buyer', 'purchaser']):
        if 'customers' in nested_data and isinstance(nested_data['customers'], list):
            results = apply_intelligent_filters(nested_data['customers'], prompt, requirements)

    elif any(keyword in prompt_lower for keyword in ['product', 'item', 'inventory', 'shoe', 'boot', 'sneaker']):
        if 'products' in nested_data and isinstance(nested_data['products'], list):
            results = apply_intelligent_filters(nested_data['products'], prompt, requirements)

    elif any(keyword in prompt_lower for keyword in ['store', 'shop', 'hour', 'open', 'close', 'address']):
        if 'store_info' in nested_data and isinstance(nested_data['store_info'], dict):
            # For store info, we return the whole object since it's a single record
            results = [nested_data['store_info']]

    return results

def get_available_databases(user_id):
    """Get all databases available for a user with improved wide database support"""
    user_databases = load_data(USER_DATABASES_FILE)
    available_dbs = []

    for db_id, db_data in user_databases.items():
        if db_data['user_id'] == user_id:
            data = db_data.get('data', {})

            # Calculate total record count for wide databases
            total_records = 0
            if isinstance(data, dict):
                # Count records in all tables
                for key, value in data.items():
                    if isinstance(value, list):
                        total_records += len(value)
                    elif isinstance(value, dict) and key not in ['store_info', 'business_info']:
                        # Count individual objects as records
                        total_records += 1
            elif isinstance(data, list):
                total_records = len(data)
            else:
                total_records = 1

            db_info = {
                'id': db_id,
                'name': db_data['name'],
                'description': db_data.get('description', ''),
                'type': db_data['file_type'],
                'record_count': total_records,
                'fields': analyze_database_structure(data),
                'data': data,
                'structure_type': 'wide' if isinstance(data, dict) and any(isinstance(v, list) for v in data.values()) else 'flat'
            }
            available_dbs.append(db_info)

    return available_dbs

def prepare_enhanced_context_with_results(prompt, business_context, available_databases, query_results):
    """Prepare context that understands wide database structures"""

    # Build results context
    results_context = "QUERY RESULTS:\n"
    if query_results:
        for db_name, results in query_results.items():
            results_context += f"\n--- {db_name} Results ---\n"
            if results:
                for i, record in enumerate(results):
                    if isinstance(record, dict):
                        # Format wide database results nicely
                        formatted_record = format_wide_record(record)
                        results_context += f"Record {i+1}:\n{formatted_record}\n"
                    else:
                        results_context += f"Record {i+1}: {record}\n"
            else:
                results_context += "No matching records found in this database.\n"
    else:
        results_context += "No query results returned. This could mean:\n"
        results_context += "- The database structure doesn't match the query\n"
        results_context += "- No data matches the search criteria\n"
        results_context += "- The database might be empty\n"

    # Build database info context that explains wide structures
    db_info_context = "DATABASE STRUCTURES AVAILABLE:\n"
    for db in available_databases:
        db_info_context += f"\n📁 {db['name']}: {db.get('description', 'No description')}\n"

        if db.get('structure_type') == 'wide':
            db_info_context += "  Structure: Wide database (multiple tables in one file)\n"
            for table_name, table_info in db.get('fields', {}).items():
                if table_info.get('type') == 'table':
                    db_info_context += f"  • {table_name}: {table_info.get('record_count', 0)} records\n"
                elif table_info.get('type') == 'object':
                    db_info_context += f"  • {table_name}: Configuration object\n"
        else:
            db_info_context += f"  Structure: Flat database ({db['record_count']} records)\n"
            if db.get('fields'):
                db_info_context += f"  Fields: {', '.join(list(db['fields'].keys())[:5])}"
                if len(db['fields']) > 5:
                    db_info_context += "..."

    system_message = f"""You are an AI assistant with access to business databases.

BUSINESS CONTEXT:
{business_context}

{db_info_context}

{results_context}

IMPORTANT: The database uses a WIDE structure with multiple tables in one file.
When responding, specify which table the data comes from (e.g., "From the customers table:").

INSTRUCTIONS:
1. Analyze the actual query results above
2. Reference specific data from the results
3. If results exist, present them clearly
4. Do not hulicinate and make sure you dont act like a bot"""

    user_message = f"""USER QUERY: {prompt}

Based on the database query results above, please provide a helpful response:"""

    return {
        'system_message': system_message,
        'user_message': user_message
    }

def format_wide_record(record):
    """Format wide database records for better readability"""
    formatted = ""
    for key, value in record.items():
        if isinstance(value, (str, int, float, bool)):
            formatted += f"  {key}: {value}\n"
        elif isinstance(value, dict):
            formatted += f"  {key}:\n"
            for sub_key, sub_value in value.items():
                formatted += f"    {sub_key}: {sub_value}\n"
        elif isinstance(value, list):
            formatted += f"  {key}: {len(value)} items\n"
            for i, item in enumerate(value[:3]):  # Show first 3 items
                if isinstance(item, dict):
                    formatted += f"    Item {i+1}: {json.dumps(item, indent=4)}\n"
                else:
                    formatted += f"    Item {i+1}: {item}\n"
            if len(value) > 3:
                formatted += f"    ... and {len(value) - 3} more items\n"
    return formatted

def prepare_enhanced_context(prompt, business_context, ai_capabilities, available_databases, conversation_context):
    """Backward compatibility wrapper"""
    return prepare_enhanced_context_with_results(
        prompt,
        business_context,
        available_databases,
        automatically_execute_query(available_databases, analyze_prompt_for_data(prompt), prompt)
    )

def build_intelligent_database_context(available_databases, requirements):
    """Build context with relevant database information"""
    if not available_databases:
        return "No databases available."

    db_context = "AVAILABLE DATABASES:\n"

    for db in available_databases:
        db_context += f"\n--- {db['name']} ---\n"
        db_context += f"Description: {db.get('description', 'No description')}\n"
        db_context += f"Records: {db['record_count']}\n"
        db_context += f"Fields: {', '.join(db['fields'].keys()) if db['fields'] else 'No fields'}\n"

        # Show sample data if relevant to the query
        if requirements['needs_query'] and any(table in db['name'].lower() for table in requirements['likely_tables']):
            sample_data = db.get('data', [])
            if isinstance(sample_data, list) and len(sample_data) > 0:
                db_context += "Sample Data:\n"
                for i, record in enumerate(sample_data[:3]):  # Show 3 sample records
                    if isinstance(record, dict):
                        # Show only relevant fields if specified
                        if requirements['expected_fields']:
                            filtered_record = {}
                            for field in requirements['expected_fields']:
                                if field in record:
                                    filtered_record[field] = record[field]
                            if filtered_record:
                                db_context += f"Record {i+1}: {json.dumps(filtered_record, indent=2)}\n"
                        else:
                            # Show first few fields
                            sample_record = {}
                            for j, (key, value) in enumerate(list(record.items())[:4]):
                                sample_record[key] = value
                            db_context += f"Record {i+1}: {json.dumps(sample_record, indent=2)}\n"

    return db_context

def transform_database_data(data, transformations):
    """n8n-style data transformation"""
    if not isinstance(data, list):
        return data

    transformed_data = []
    for item in data:
        if isinstance(item, dict):
            transformed_item = item.copy()

            # Apply transformations
            for field, transformation in transformations.items():
                if field in transformed_item:
                    transformed_item[field] = apply_transformation(
                        transformed_item[field],
                        transformation
                    )

            transformed_data.append(transformed_item)

    return transformed_data

def apply_transformation(value, transformation):
    """Apply n8n-style transformations"""
    if transformation == 'uppercase':
        return str(value).upper()
    elif transformation == 'lowercase':
        return str(value).lower()
    elif transformation == 'trim':
        return str(value).strip()
    elif transformation.startswith('prefix:'):
        prefix = transformation.split(':', 1)[1]
        return f"{prefix}{value}"
    elif transformation.startswith('suffix:'):
        suffix = transformation.split(':', 1)[1]
        return f"{value}{suffix}"
    # Add more transformations as needed

    return value

# Generate AI response using OpenRouter API
def generate_ai_response(prompt, user_id, ai_id, conversation_context=None):
    api_key = get_user_api_key(user_id, "OpenRouter")
    if not api_key:
        return "Error: No OpenRouter API key configured. Please add your API key in the API Settings."

    # Get user's AI configuration
    ai_workforce = load_data(AI_WORKFORCE_FILE)

    if ai_id not in ai_workforce:
        return "Error: AI configuration not found."

    ai_config = ai_workforce[ai_id]
    business_context = ai_config.get('business_info', 'No business context provided.')

    # Get user's databases
    available_databases = get_available_databases(user_id)

    # Analyze prompt for data requirements
    data_requirements = analyze_prompt_for_data(prompt)

    # EXECUTE THE QUERY AUTOMATICALLY (This is the key fix!)
    query_results = automatically_execute_query(available_databases, data_requirements, prompt)

    # Prepare enhanced context with actual query results
    context = prepare_enhanced_context_with_results(prompt, business_context, available_databases, query_results)

    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        messages = [
            {"role": "system", "content": context['system_message']},
        ]

        if conversation_context:
            for msg in conversation_context:
                messages.append(msg)

        messages.append({"role": "user", "content": context['user_message']})

        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000  # Increased for data-rich responses
        }

        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=45
        )

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error: API request failed with status {response.status_code}. Response: {response.text}"

    except Exception as e:
        return f"Error generating response: {str(e)}"

def automatically_execute_query(available_databases, requirements, prompt):
    """Automatically execute queries based on prompt analysis"""
    results = {}

    for db in available_databases:
        db_name = db['name'].lower()
        data = db.get('data', [])

        if not isinstance(data, list) or len(data) == 0:
            continue

        # Check if this database matches the query requirements
        if any(keyword in db_name for keyword in requirements['likely_tables']):
            filtered_data = apply_intelligent_filters(data, prompt, requirements)

            if filtered_data:
                # Limit to reasonable number of results
                results[db['name']] = filtered_data[:10]  # Show max 10 records

    return results

def apply_intelligent_filters(data, prompt, requirements):
    """Apply intelligent filters based on prompt content - IMPROVED"""
    if not isinstance(data, list):
        return []

    prompt_lower = prompt.lower()
    filtered_data = data.copy()

    # Apply field-based filtering if specific fields requested
    if requirements['expected_fields']:
        filtered_data = [
            {k: v for k, v in record.items() if k in requirements['expected_fields'] or k in list(record.keys())[:3]}
            for record in filtered_data
            if isinstance(record, dict)
        ]

    # Apply content-based filters from prompt
    if 'recent' in prompt_lower or 'last' in prompt_lower:
        filtered_data = filter_by_date(filtered_data, prompt)

    if any(word in prompt_lower for word in ['under', 'less than', 'below', 'cheap']):
        filtered_data = filter_by_numeric(filtered_data, prompt, 'less')

    if any(word in prompt_lower for word in ['over', 'greater than', 'above', 'expensive']):
        filtered_data = filter_by_numeric(filtered_data, prompt, 'greater')

    # Filter for specific names if mentioned
    name_matches = filter_by_name(filtered_data, prompt)
    if name_matches:
        filtered_data = name_matches

    return filtered_data

def filter_by_name(data, prompt):
    """Filter data by name mentions in prompt"""
    import re

    # Look for quoted names or specific name patterns
    quoted_names = re.findall(r'"([^"]*)"', prompt)
    if quoted_names:
        return [record for record in data
                if isinstance(record, dict) and 'name' in record
                and any(q_name.lower() in record['name'].lower() for q_name in quoted_names)]

    # Look for name-like patterns (capitalized words)
    name_patterns = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', prompt)
    if name_patterns:
        return [record for record in data
                if isinstance(record, dict) and 'name' in record
                and any(pattern.lower() in record['name'].lower() for pattern in name_patterns)]

    return None

def filter_by_date(data, prompt):
    """Filter data based on date references in prompt"""
    prompt_lower = prompt.lower()
    filtered_data = []

    date_fields = ['date', 'created', 'timestamp', 'time', 'purchase_date']

    for record in data:
        if isinstance(record, dict):
            for field in date_fields:
                if field in record and isinstance(record[field], str):
                    # Simple date filtering logic
                    if 'today' in prompt_lower:
                        # Implement today's date logic if needed
                        filtered_data.append(record)
                    elif 'yesterday' in prompt_lower:
                        # Implement yesterday's date logic
                        filtered_data.append(record)
                    else:
                        # Default: include if date field exists
                        filtered_data.append(record)
                    break
            else:
                # No date field found, include anyway
                filtered_data.append(record)

    return filtered_data

def filter_by_numeric(data, prompt, comparison_type):
    """Filter data based on numeric comparisons"""
    import re

    # Extract numbers from prompt
    numbers = re.findall(r'\$?(\d+)', prompt)
    if not numbers:
        return data

    target_value = float(numbers[0])
    filtered_data = []

    numeric_fields = ['price', 'amount', 'total', 'cost', 'quantity']

    for record in data:
        if isinstance(record, dict):
            for field in numeric_fields:
                if field in record:
                    try:
                        value = float(record[field])
                        if comparison_type == 'less' and value < target_value:
                            filtered_data.append(record)
                        elif comparison_type == 'greater' and value > target_value:
                            filtered_data.append(record)
                        break
                    except (ValueError, TypeError):
                        continue

    return filtered_data if filtered_data else data

def save_facebook_token(user_id, ai_id, access_token, page_id=None, page_name=None):
    """Save Facebook page access token to database"""
    facebook_tokens = load_data(FACEBOOK_BOTS_FILE)

    token_id = f"{user_id}_{ai_id}"
    facebook_tokens[token_id] = {
        'user_id': user_id,
        'ai_id': ai_id,
        'access_token': access_token,
        'page_id': page_id,
        'page_name': page_name,
        'created_at': datetime.now().isoformat()
    }

    save_data(facebook_tokens, FACEBOOK_BOTS_FILE)
    return True

def has_facebook_token(user_id, ai_id):
    """Check if user has Facebook token"""
    facebook_tokens = load_data(FACEBOOK_BOTS_FILE)
    token_id = f"{user_id}_{ai_id}"
    return token_id in facebook_tokens

def get_facebook_token(user_id, ai_id):
    """Retrieve Facebook access token"""
    facebook_tokens = load_data(FACEBOOK_BOTS_FILE)
    token_id = f"{user_id}_{ai_id}"
    return facebook_tokens.get(token_id)



def post_to_facebook_page(page_id, access_token, message, link=None):
    """Post to a Facebook page"""
    try:
        url = f"https://graph.facebook.com/v13.0/{page_id}/feed"

        payload = {
            'message': message,
            'access_token': access_token
        }

        if link:
            payload['link'] = link

        response = requests.post(url, data=payload)

        if response.status_code == 200:
            return response.json()

        return None

    except Exception as e:
        print(f"Error posting to Facebook: {e}")
        return None

def get_page_insights(page_id, access_token, metric='page_engaged_users'):
    """Get Facebook page insights"""
    try:
        url = f"https://graph.facebook.com/v13.0/{page_id}/insights"
        params = {
            'metric': metric,
            'access_token': access_token,
            'period': 'day'
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.json()

        return None

    except Exception as e:
        print(f"Error getting Facebook insights: {e}")
        return None

def respond_to_comment(comment_id, access_token, message):
    """Respond to a Facebook comment"""
    try:
        url = f"https://graph.facebook.com/v13.0/{comment_id}/comments"

        payload = {
            'message': message,
            'access_token': access_token
        }

        response = requests.post(url, data=payload)

        if response.status_code == 200:
            return response.json()

        return None

    except Exception as e:
        print(f"Error responding to comment: {e}")
        return None

def process_facebook_change(change, user_id, ai_id):
    """Process Facebook real-time updates with auto-reply"""
    change_type = change.get('field')
    change_value = change.get('value')

    # Get Facebook access token
    access_token = get_facebook_token(user_id, ai_id)
    if not access_token:
        return

    if change_type == 'feed':
        # New post on page
        item = change_value.get('item')
        if item == 'post' and change_value.get('verb') == 'add':
            post_id = change_value.get('post_id')
            # You could automatically respond to new posts here

    elif change_type == 'comments':
        # New comment - AUTO-REPLY FUNCTIONALITY
        if change_value.get('verb') == 'add':
            comment_id = change_value.get('comment_id')

            # Get comment details
            comment_url = f"https://graph.facebook.com/v13.0/{comment_id}?fields=message,from,post&access_token={access_token}"
            comment_response = requests.get(comment_url)

            if comment_response.status_code == 200:
                comment_data = comment_response.json()
                comment_message = comment_data.get('message', '')
                commenter_name = comment_data.get('from', {}).get('name', 'Unknown')
                post_id = comment_data.get('post', {}).get('id', '')

                # Get auto-reply rules
                auto_reply_rules = get_auto_reply_rules(user_id, ai_id)

                # Check if we should auto-reply
                if should_auto_reply(comment_message, auto_reply_rules):
                    # Get business context for personalized response
                    ai_workforce = load_data(AI_WORKFORCE_FILE)
                    ai_config = ai_workforce.get(ai_id, {})
                    business_context = ai_config.get('business_info', '')

                    # Generate or use template response
                    ai_response = generate_auto_reply(comment_message, auto_reply_rules, business_context)

                    # Respond to comment
                    respond_to_comment(comment_id, access_token, ai_response)

                    # Save to chat history
                    save_chat_history(
                        user_id,
                        ai_id,
                        'facebook_auto_reply',
                        comment_id,
                        f"Auto-reply to {commenter_name}: {comment_message}",
                        ai_response,
                        "outgoing"
                    )
                else:
                    # Regular AI response for non-auto-reply comments
                    ai_response = generate_ai_response(
                        f"Facebook comment from {commenter_name}: {comment_message}",
                        user_id,
                        ai_id
                    )

                    # Respond to comment
                    respond_to_comment(comment_id, access_token, ai_response)

                    # Save to chat history
                    save_chat_history(
                        user_id,
                        ai_id,
                        'facebook_comment',
                        comment_id,
                        comment_message,
                        ai_response,
                        "incoming"
                    )

# Database upload functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def search_user_databases(user_id, query, max_results=5):
    """Search through user's uploaded databases"""
    user_databases = load_data(USER_DATABASES_FILE)
    results = []

    for db_id, db_data in user_databases.items():
        if db_data['user_id'] == user_id and db_data.get('data'):
            data = db_data['data']

            if isinstance(data, list):  # Array of objects
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        # Search through all values in the object
                        for key, value in item.items():
                            if query.lower() in str(value).lower():
                                results.append({
                                    'database': db_data['name'],
                                    'record_index': i,
                                    'field': key,
                                    'value': value,
                                    'full_record': item
                                })
                                if len(results) >= max_results:
                                    return results
            elif isinstance(data, dict):  # Single object
                for key, value in data.items():
                    if query.lower() in str(value).lower():
                        results.append({
                            'database': db_data['name'],
                            'field': key,
                            'value': value,
                            'full_record': data
                        })
                        if len(results) >= max_results:
                            return results

    return results

# Add these functions with your other utility functions

def save_scheduled_post(user_id, ai_id, platform, content, schedule_time, post_type):
    """Save a scheduled post to the database"""
    scheduled_posts = load_data(SCHEDULED_POSTS_FILE)

    post_id = str(uuid.uuid4())
    scheduled_posts[post_id] = {
        'id': post_id,
        'user_id': user_id,
        'ai_id': ai_id,
        'platform': platform,
        'content': content,
        'post_type': post_type,
        'scheduled_time': schedule_time,
        'status': 'scheduled',
        'created_at': datetime.now().isoformat()
    }

    save_data(scheduled_posts, SCHEDULED_POSTS_FILE)
    return post_id

def get_scheduled_posts(user_id, ai_id=None):
    """Get scheduled posts for a user"""
    scheduled_posts = load_data(SCHEDULED_POSTS_FILE)
    user_posts = {}

    for post_id, post_data in scheduled_posts.items():
        if post_data['user_id'] == user_id:
            if ai_id is None or post_data['ai_id'] == ai_id:
                user_posts[post_id] = post_data

    return user_posts

def save_content_strategy(user_id, ai_id, strategy_data):
    """Save content strategy for automated posting"""
    content_strategies = load_data(CONTENT_STRATEGY_FILE)

    strategy_id = f"{user_id}_{ai_id}"
    content_strategies[strategy_id] = {
        'user_id': user_id,
        'ai_id': ai_id,
        **strategy_data,
        'updated_at': datetime.now().isoformat()
    }

    save_data(content_strategies, CONTENT_STRATEGY_FILE)
    return strategy_id

def get_content_strategy(user_id, ai_id):
    """Get content strategy for an AI"""
    content_strategies = load_data(CONTENT_STRATEGY_FILE)
    strategy_id = f"{user_id}_{ai_id}"
    return content_strategies.get(strategy_id, {})

def save_auto_reply_rule(user_id, ai_id, rule_data):
    """Save auto-reply rules"""
    auto_reply_rules = load_data(AUTO_REPLY_RULES_FILE)

    rule_id = str(uuid.uuid4())
    auto_reply_rules[rule_id] = {
        'id': rule_id,
        'user_id': user_id,
        'ai_id': ai_id,
        **rule_data,
        'created_at': datetime.now().isoformat()
    }

    save_data(auto_reply_rules, AUTO_REPLY_RULES_FILE)
    return rule_id

def get_auto_reply_rules(user_id, ai_id):
    """Get auto-reply rules for an AI"""
    auto_reply_rules = load_data(AUTO_REPLY_RULES_FILE)
    user_rules = {}

    for rule_id, rule_data in auto_reply_rules.items():
        if rule_data['user_id'] == user_id and rule_data['ai_id'] == ai_id:
            user_rules[rule_id] = rule_data

    return user_rules

def analyze_content_performance(access_token, page_id, days=7):
    """Analyze content performance with enhanced metrics"""
    try:
        # Get page posts
        posts_url = f"https://graph.facebook.com/v13.0/{page_id}/posts"
        params = {
            'access_token': access_token,
            'fields': 'message,created_time,likes.summary(true),comments.summary(true),shares',
            'limit': 50
        }

        response = requests.get(posts_url, params=params)

        if response.status_code == 200:
            posts_data = response.json().get('data', [])

            # Analyze engagement
            analysis = {
                'total_posts': len(posts_data),
                'total_likes': 0,
                'total_comments': 0,
                'total_shares': 0,
                'avg_engagement_rate': 0,
                'top_performing_posts': [],
                'best_posting_times': {},
                'content_type_analysis': {}
            }

            for post in posts_data:
                likes = post.get('likes', {}).get('summary', {}).get('total_count', 0)
                comments = post.get('comments', {}).get('summary', {}).get('total_count', 0)
                shares = post.get('shares', {}).get('count', 0)

                analysis['total_likes'] += likes
                analysis['total_comments'] += comments
                analysis['total_shares'] += shares

                # Track top performing posts
                engagement = likes + comments * 2 + shares * 3
                if engagement > 0:
                    analysis['top_performing_posts'].append({
                        'message': post.get('message', '')[:100] + '...',
                        'engagement': engagement,
                        'time': post.get('created_time', '')
                    })

            # Sort top posts by engagement
            analysis['top_performing_posts'].sort(key=lambda x: x['engagement'], reverse=True)
            analysis['top_performing_posts'] = analysis['top_performing_posts'][:5]

            # Calculate averages
            if analysis['total_posts'] > 0:
                analysis['avg_engagement_rate'] = (
                    (analysis['total_likes'] + analysis['total_comments'] + analysis['total_shares']) /
                    analysis['total_posts']
                )

            return analysis
        else:
            return {'error': f'API error: {response.status_code}'}

    except Exception as e:
        return {'error': str(e)}

def generate_content_ideas(business_context, topic, count=5):
    """Generate content ideas using AI"""
    prompt = f"""Generate {count} engaging content ideas for {business_context} about {topic}.
    For each idea, provide:
    1. A catchy title
    2. Main content points
    3. Suggested hashtags
    4. Best time to post

    Make the ideas creative and engaging:"""

    # This would use your existing AI response generation
    return generate_ai_response(prompt, "system", "content_generator")

def should_auto_reply(comment_text, rules):
    """Check if a comment should trigger auto-reply based on rules"""
    comment_lower = comment_text.lower()

    for rule_id, rule in rules.items():
        # Keyword matching
        if 'keywords' in rule:
            if any(keyword.lower() in comment_lower for keyword in rule['keywords']):
                return True

        # Sentiment analysis (simple version)
        if 'sentiment' in rule:
            positive_words = ['great', 'awesome', 'love', 'amazing', 'excellent']
            negative_words = ['bad', 'terrible', 'hate', 'awful', 'disappointing']

            if rule['sentiment'] == 'positive':
                if any(word in comment_lower for word in positive_words):
                    return True
            elif rule['sentiment'] == 'negative':
                if any(word in comment_lower for word in negative_words):
                    return True

    return False

def generate_auto_reply(comment_text, rules, business_context):
    """Generate context-aware auto-reply"""
    for rule_id, rule in rules.items():
        if 'response_template' in rule:
            # Use template if available
            return rule['response_template'].format(
                business_name=business_context,
                comment=comment_text
            )

    # Fallback to AI-generated response
    prompt = f"""Generate a friendly and professional response to this comment: "{comment_text}"
    for {business_context}. Keep it engaging and appropriate:"""

    return generate_ai_response(prompt, "system", "auto_reply")

# Template context processor
@app.context_processor
def utility_processor():
    def has_facebook_token_context(user_id, ai_id):
        return has_facebook_token(user_id, ai_id)
    return dict(has_facebook_token=has_facebook_token_context)

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    return f"Internal Server Error: {str(error)}", 500

@app.errorhandler(404)
def not_found_error(error):
    return "Page not found", 404

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        users = load_data(USERS_FILE)

        if email in users and check_password_hash(users[email]['password'], password):
            session['user_id'] = users[email]['id']
            session['user_email'] = email
            session['business_name'] = users[email]['business_name']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')

    return render_template_string(LOGIN_TEMPLATE)

# Add these routes after your existing routes

@app.route('/api/database/query', methods=['POST'])
@login_required
def api_database_query():
    """n8n-style API endpoint for database queries"""
    try:
        data = request.get_json()
        user_id = session['user_id']

        # Extract query parameters (like n8n's HTTP node)
        query_params = {
            'filters': data.get('filters', {}),
            'sort': data.get('sort', {}),
            'pagination': data.get('pagination', {}),
            'database': data.get('database')
        }

        results = query_user_database(user_id, query_params)

        return jsonify({
            'success': True,
            'data': results,
            'total': len(results),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400

@app.route('/api/ai/query', methods=['POST'])
@login_required
def ai_data_query():
    """Endpoint for AI to query user data with enhanced context"""
    try:
        data = request.get_json()
        user_id = session['user_id']
        ai_id = data.get('ai_id')
        query = data.get('query', '')

        # Get available databases
        available_databases = get_available_databases(user_id)

        # Generate response with enhanced data access
        ai_workforce = load_data(AI_WORKFORCE_FILE)
        ai_config = ai_workforce.get(ai_id, {})
        business_context = ai_config.get('business_info', '')

        context = prepare_enhanced_context(query, business_context, {}, available_databases, None)

        # Use the enhanced generate_ai_response function
        response = generate_ai_response(query, user_id, ai_id)

        return jsonify({
            'success': True,
            'response': response,
            'databases_accessed': [db['name'] for db in available_databases],
            'data_requirements': context['data_requirements'],
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        business_name = request.form['business_name']

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template_string(SIGNUP_TEMPLATE)

        users = load_data(USERS_FILE)

        if email in users:
            flash('Email already registered', 'error')
            return render_template_string(SIGNUP_TEMPLATE)

        user_id = str(uuid.uuid4())
        users[email] = {
            'id': user_id,
            'email': email,
            'password': generate_password_hash(password),
            'business_name': business_name,
            'created_at': datetime.now().isoformat()
        }

        save_data(users, USERS_FILE)

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template_string(SIGNUP_TEMPLATE)

@app.route('/dashboard')
@login_required
def dashboard():
    # Load AI workforce data
    ai_workforce = load_data(AI_WORKFORCE_FILE)
    user_ai = [ai for ai in ai_workforce.values() if ai['user_id'] == session['user_id']]

    # Load user databases
    user_databases = load_data(USER_DATABASES_FILE)
    user_db_count = len([db for db in user_databases.values() if db['user_id'] == session['user_id']])

    # Load chat history
    chat_history = load_data(CHAT_HISTORY_FILE)
    user_chat_count = len([conv for conv in chat_history.values() if conv['user_id'] == session['user_id']])

    # Count active AI agents
    active_ai_count = len([ai for ai in user_ai if ai.get('status') == 'active'])

    # Get Facebook token status for each AI
    facebook_tokens = load_data(FACEBOOK_BOTS_FILE)
    for ai in user_ai:
        token_id = f"{session['user_id']}_{ai['id']}"
        ai['has_facebook_token'] = token_id in facebook_tokens

        # Generate webhook URL for each AI
        ai['webhook_url'] = f"{request.host_url}webhook/{ai['id']}"

    return render_template_string(DASHBOARD_TEMPLATE,
                                 user_ai=user_ai,
                                 available_ai=AVAILABLE_AI_SERVICES,
                                 business_name=session.get('business_name', 'Your Business'),
                                 user_email=session.get('user_email', 'User'),
                                 user_db_count=user_db_count,
                                 user_chat_count=user_chat_count,
                                 active_ai_count=active_ai_count)

@app.route('/add_ai/<ai_id>')
@login_required
def add_ai(ai_id):
    ai_service = next((ai for ai in AVAILABLE_AI_SERVICES if ai['id'] == ai_id), None)
    if not ai_service:
        flash('AI service not found', 'error')
        return redirect(url_for('dashboard'))
    return render_template_string(ADD_AI_TEMPLATE, ai_service=ai_service)

@app.route('/save_ai', methods=['POST'])
@login_required
def save_ai():
    ai_id = request.form['ai_id']
    ai_name = request.form['ai_name']
    description = request.form['description']
    business_info = request.form['business_info']

    ai_workforce = load_data(AI_WORKFORCE_FILE)
    ai_service = next((ai for ai in AVAILABLE_AI_SERVICES if ai['id'] == ai_id), None)

    if not ai_service:
        flash('AI service not found', 'error')
        return redirect(url_for('dashboard'))

    ai_entry = {
        'id': str(uuid.uuid4()),
        'user_id': session['user_id'],
        'ai_id': ai_id,
        'name': ai_name,
        'description': description,
        'business_info': business_info,
        'service_name': ai_service['name'],
        'status': 'active',
        'created_at': datetime.now().isoformat()
    }

    ai_workforce[ai_entry['id']] = ai_entry
    save_data(ai_workforce, AI_WORKFORCE_FILE)
    flash(f'{ai_service["name"]} added successfully!', 'success')
    return redirect(url_for('dashboard'))

# Replace your current routes with these simplified versions

@app.route('/api_settings')
@login_required
def api_settings():
    try:
        api_keys = load_data(API_KEYS_FILE)
        user_api_keys = [key for key in api_keys.values() if key.get('user_id') == session.get('user_id')]
        return render_template_string(API_SETTINGS_TEMPLATE, api_keys=user_api_keys)
    except Exception as e:
        return f"Error loading API settings: {str(e)}", 500

@app.route('/chat_history')
@login_required
def chat_history():
    try:
        ai_id = request.args.get('ai_id')
        platform = request.args.get('platform')
        chats = get_user_chat_history(session.get('user_id'), ai_id, platform)

        # Load AI workforce for display names
        ai_workforce = load_data(AI_WORKFORCE_FILE)
        user_ai = {}
        for ai_id, ai_data in ai_workforce.items():
            if ai_data.get('user_id') == session.get('user_id'):
                user_ai[ai_id] = ai_data

        return render_template_string(CHAT_HISTORY_TEMPLATE, chats=chats, ai_workforce=user_ai)
    except Exception as e:
        return f"Error loading chat history: {str(e)}", 500

@app.route('/save_api_key', methods=['POST'])
@login_required
def save_api_key():
    service_name = request.form['service_name']
    api_key = request.form['api_key']

    api_keys = load_data(API_KEYS_FILE)

    for key_id, key_data in api_keys.items():
        if key_data['user_id'] == session['user_id'] and key_data['service_name'] == service_name:
            key_data['api_key'] = api_key
            key_data['updated_at'] = datetime.now().isoformat()
            save_data(api_keys, API_KEYS_FILE)
            flash('API key updated successfully!', 'success')
            return redirect(url_for('api_settings'))

    new_key = {
        'id': str(uuid.uuid4()),
        'user_id': session['user_id'],
        'service_name': service_name,
        'api_key': api_key,
        'created_at': datetime.now().isoformat()
    }

    api_keys[new_key['id']] = new_key
    save_data(api_keys, API_KEYS_FILE)
    flash('API key saved successfully!', 'success')
    return redirect(url_for('api_settings'))

@app.route('/delete_api_key/<key_id>')
@login_required
def delete_api_key(key_id):
    api_keys = load_data(API_KEYS_FILE)
    if key_id in api_keys and api_keys[key_id]['user_id'] == session['user_id']:
        del api_keys[key_id]
        save_data(api_keys, API_KEYS_FILE)
        flash('API key deleted successfully!', 'success')
    else:
        flash('API key not found', 'error')
    return redirect(url_for('api_settings'))

@app.route('/delete_ai/<ai_id>')
@login_required
def delete_ai(ai_id):
    ai_workforce = load_data(AI_WORKFORCE_FILE)
    if ai_id in ai_workforce and ai_workforce[ai_id]['user_id'] == session['user_id']:
        del ai_workforce[ai_id]
        save_data(ai_workforce, AI_WORKFORCE_FILE)
        flash('AI deleted successfully!', 'success')
    else:
        flash('AI not found', 'error')
    return redirect(url_for('dashboard'))


@app.route('/setup_telegram_bot', methods=['POST'])
@login_required
def setup_telegram_bot():
    bot_token = request.form['bot_token']
    ai_id = request.form['ai_id']

    try:
        test_url = f"https://api.telegram.org/bot{bot_token}/getMe"
        response = requests.get(test_url, timeout=10)
        if response.status_code != 200:
            flash('Invalid Telegram bot token', 'error')
            return redirect(url_for('dashboard'))

        bot_info = response.json()
        bot_username = bot_info['result']['username']

        telegram_bots = load_data(TELEGRAM_BOTS_FILE)

        for bot_id, bot_data in telegram_bots.items():
            if bot_data['user_id'] == session['user_id'] and bot_data['ai_id'] == ai_id:
                bot_data['bot_token'] = bot_token
                bot_data['bot_username'] = bot_username
                bot_data['updated_at'] = datetime.now().isoformat()
                save_data(telegram_bots, TELEGRAM_BOTS_FILE)
                flash('Telegram bot updated successfully!', 'success')
                return redirect(url_for('dashboard'))

        new_bot = {
            'id': str(uuid.uuid4()),
            'user_id': session['user_id'],
            'ai_id': ai_id,
            'bot_token': bot_token,
            'bot_username': bot_username,
            'created_at': datetime.now().isoformat()
        }

        telegram_bots[new_bot['id']] = new_bot
        save_data(telegram_bots, TELEGRAM_BOTS_FILE)

        webhook_url = f"{request.host_url}webhook/telegram/{new_bot['id']}"
        set_webhook_url = f"https://api.telegram.org/bot{bot_token}/setWebhook?url={webhook_url}"

        webhook_response = requests.get(set_webhook_url, timeout=10)
        if webhook_response.status_code == 200:
            flash('Telegram bot configured successfully! Webhook set.', 'success')
        else:
            flash('Telegram bot configured but failed to set webhook.', 'warning')

        return redirect(url_for('dashboard'))

    except Exception as e:
        flash(f'Error configuring Telegram bot: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/setup_whatsapp_bot', methods=['POST'])
@login_required
def setup_whatsapp_bot():
    provider = request.form['provider']
    phone_number = request.form['phone_number']
    api_key = request.form['api_key']
    auth_token = request.form.get('auth_token', '')
    whatsapp_number = request.form.get('whatsapp_number', '')
    ai_id = request.form['ai_id']

    whatsapp_bots = load_data(WHATSAPP_BOTS_FILE)

    for bot_id, bot_data in whatsapp_bots.items():
        if bot_data['user_id'] == session['user_id'] and bot_data['ai_id'] == ai_id:
            bot_data.update({
                'provider': provider,
                'phone_number': phone_number,
                'api_key': api_key,
                'auth_token': auth_token,
                'whatsapp_number': whatsapp_number,
                'updated_at': datetime.now().isoformat()
            })
            save_data(whatsapp_bots, WHATSAPP_BOTS_FILE)
            flash('WhatsApp bot updated successfully!', 'success')
            return redirect(url_for('dashboard'))

    new_bot = {
        'id': str(uuid.uuid4()),
        'user_id': session['user_id'],
        'ai_id': ai_id,
        'provider': provider,
        'phone_number': phone_number,
        'api_key': api_key,
        'auth_token': auth_token,
        'whatsapp_number': whatsapp_number,
        'created_at': datetime.now().isoformat()
    }

    whatsapp_bots[new_bot['id']] = new_bot
    save_data(whatsapp_bots, WHATSAPP_BOTS_FILE)
    flash('WhatsApp bot configured successfully!', 'success')
    return redirect(url_for('dashboard'))

# Webhook routes
@app.route('/webhook/<ai_id>', methods=['POST'])
def ai_webhook(ai_id):
    ai_workforce = load_data(AI_WORKFORCE_FILE)
    if ai_id not in ai_workforce:
        return jsonify({'error': 'AI not found'}), 404

    ai_config = ai_workforce[ai_id]
    user_id = ai_config['user_id']

    if ai_config['ai_id'] == 'telegram':
        return process_telegram_webhook(request.json, user_id, ai_id)
    elif ai_config['ai_id'] == 'whatsapp':
        return process_telegram_webhook(request.json, user_id, ai_id)
    else:
        return process_generic_webhook(request.json, user_id, ai_id)

@app.route('/webhook/telegram/<bot_id>', methods=['POST'])
def telegram_webhook(bot_id):
    telegram_bots = load_data(TELEGRAM_BOTS_FILE)
    if bot_id not in telegram_bots:
        return jsonify({'error': 'Bot not found'}), 404

    bot_config = telegram_bots[bot_id]
    user_id = bot_config['user_id']
    ai_id = bot_config['ai_id']
    return process_telegram_webhook(request.json, user_id, ai_id)

def process_telegram_webhook(update, user_id, ai_id):
    try:
        if 'message' not in update or 'text' not in update['message']:
            return jsonify({'status': 'ok'})

        message = update['message']
        chat_id = message['chat']['id']
        text = message['text']
        sender_id = message['from']['id']

        chat_history = get_user_chat_history(user_id, ai_id, 'telegram')
        conversation_context = []

        for conv_id, conv_data in chat_history.items():
            if str(sender_id) in conv_id:
                for msg in conv_data['messages']:
                    if msg['direction'] == 'incoming':
                        conversation_context.append({"role": "user", "content": msg['message']})
                    else:
                        conversation_context.append({"role": "assistant", "content": msg['response']})

        ai_response = generate_ai_response(text, user_id, ai_id, conversation_context)
        save_chat_history(user_id, ai_id, 'telegram', sender_id, text, ai_response, "incoming")

        telegram_bots = load_data(TELEGRAM_BOTS_FILE)
        bot_token = None

        for bot_id, bot_data in telegram_bots.items():
            if bot_data['user_id'] == user_id and bot_data['ai_id'] == ai_id:
                bot_token = bot_data['bot_token']
                break

        if not bot_token:
            return jsonify({'error': 'Bot token not found'}), 500

        send_message_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {'chat_id': chat_id, 'text': ai_response}

        response = requests.post(send_message_url, json=payload, timeout=10)
        if response.status_code == 200:
            save_chat_history(user_id, ai_id, 'telegram', sender_id, ai_response, "", "outgoing")

        return jsonify({'status': 'ok'})

    except Exception as e:
        print(f"Error processing Telegram webhook: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/webhook/whatsapp/<ai_id>', methods=['POST'])
def whatsapp_webhook(ai_id):
    try:
        ai_workforce = load_data(AI_WORKFORCE_FILE)
        if ai_id not in ai_workforce:
            return jsonify({'error': 'AI not found'}), 404

        ai_config = ai_workforce[ai_id]
        user_id = ai_config['user_id']
        update = request.get_json()
        return process_whatsapp_message(update, user_id, ai_id)

    except Exception as e:
        print(f"Error in WhatsApp webhook: {e}")
        return jsonify({'error': str(e)}), 500

def process_whatsapp_message(update, user_id, ai_id):
    try:
        if 'messages' in update:
            message_data = update['messages'][0]
            sender_id = message_data['from']
            message_text = message_data['text']['body']

            chat_history = get_user_chat_history(user_id, ai_id, 'whatsapp')
            conversation_context = []

            for conv_id, conv_data in chat_history.items():
                if sender_id in conv_id:
                    for msg in conv_data['messages']:
                        if msg['direction'] == 'incoming':
                            conversation_context.append({"role": "user", "content": msg['message']})
                        else:
                            conversation_context.append({"role": "assistant", "content": msg['response']})

            ai_response = generate_ai_response(message_text, user_id, ai_id, conversation_context)
            save_chat_history(user_id, ai_id, 'whatsapp', sender_id, message_text, ai_response, "incoming")
            send_whatsapp_message(sender_id, ai_response, user_id, ai_id)

            return jsonify({'status': 'success'})

        return jsonify({'status': 'no message found'})

    except Exception as e:
        print(f"Error processing WhatsApp message: {e}")
        return jsonify({'error': str(e)}), 500

def send_whatsapp_message(recipient, message, user_id, ai_id):
    whatsapp_bots = load_data(WHATSAPP_BOTS_FILE)
    bot_config = None

    for bot_id, bot_data in whatsapp_bots.items():
        if bot_data['user_id'] == user_id and bot_data['ai_id'] == ai_id:
            bot_config = bot_data
            break

    if not bot_config:
        print("WhatsApp bot configuration not found")
        return False

    provider = bot_config.get('provider', 'generic')

    if provider == 'twilio':
        account_sid = bot_config.get('account_sid')
        auth_token = bot_config.get('auth_token')
        whatsapp_number = bot_config.get('whatsapp_number')

        if not all([account_sid, auth_token, whatsapp_number]):
            print("Twilio credentials incomplete")
            return False

        try:
            from twilio.rest import Client
            client = Client(account_sid, auth_token)
            message = client.messages.create(
                body=message,
                from_=f'whatsapp:{whatsapp_number}',
                to=f'whatsapp:{recipient}'
            )
            save_chat_history(user_id, ai_id, 'whatsapp', recipient, message, "", "outgoing")
            return True
        except Exception as e:
            print(f"Twilio error: {e}")
            return False

    elif provider == 'wati':
        api_key = bot_config.get('api_key')
        wati_url = bot_config.get('wati_url', 'https://api.wati.io')

        if not api_key:
            print("WATI API key missing")
            return False

        try:
            url = f"{wati_url}/v1/sendSessionMessage/{recipient}"
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            payload = {'text': message}

            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                save_chat_history(user_id, ai_id, 'whatsapp', recipient, message, "", "outgoing")
                return True
            else:
                print(f"WATI error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"WATI error: {e}")
            return False

    else:
        access_token = bot_config.get('access_token')
        phone_number_id = bot_config.get('phone_number_id')

        if not all([access_token, phone_number_id]):
            print("Generic WhatsApp credentials incomplete")
            return False

        try:
            url = f"https://graph.facebook.com/v13.0/{phone_number_id}/messages"
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            payload = {
                'messaging_product': 'whatsapp',
                'to': recipient,
                'text': {'body': message}
            }

            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                save_chat_history(user_id, ai_id, 'whatsapp', recipient, message, "", "outgoing")
                return True
            else:
                print(f"WhatsApp API error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"WhatsApp API error: {e}")
            return False

# Facebook routes
@app.route('/facebook/setup/<ai_id>', methods=['GET', 'POST'])
@login_required
def facebook_setup(ai_id):
    """Simple form to paste Facebook access token"""
    if request.method == 'POST':
        access_token = request.form['access_token']
        page_id = request.form.get('page_id', '').strip()
        page_name = request.form.get('page_name', '').strip()

        if not access_token:
            flash('Access token is required', 'error')
            return redirect(url_for('facebook_setup', ai_id=ai_id))

        # Save the token
        success = save_facebook_token(session['user_id'], ai_id, access_token, page_id, page_name)

        if success:
            flash('Facebook access token saved successfully!', 'success')
        else:
            flash('Failed to save access token', 'error')

        return redirect(url_for('dashboard'))

    # Check if already has token
    current_token = get_facebook_token(session['user_id'], ai_id)

    return render_template_string(FACEBOOK_SETUP_TEMPLATE,
                                ai_id=ai_id,
                                current_token=current_token)

@app.route('/facebook/delete/<ai_id>')
@login_required
def facebook_delete_token(ai_id):
    """Delete Facebook token"""
    facebook_tokens = load_data(FACEBOOK_BOTS_FILE)
    token_id = f"{session['user_id']}_{ai_id}"

    if token_id in facebook_tokens:
        del facebook_tokens[token_id]
        save_data(facebook_tokens, FACEBOOK_BOTS_FILE)
        flash('Facebook token deleted successfully!', 'success')
    else:
        flash('No Facebook token found', 'error')

    return redirect(url_for('dashboard'))

@app.route('/webhook/facebook/<ai_id>', methods=['GET', 'POST'])
def facebook_webhook(ai_id):
    if request.method == 'GET':
        # Verify webhook (keep this for Facebook validation)
        hub_mode = request.args.get('hub.mode')
        hub_token = request.args.get('hub.verify_token')
        hub_challenge = request.args.get('hub.challenge')

        if hub_mode == 'subscribe' and hub_token == FACEBOOK_VERIFY_TOKEN:
            return hub_challenge
        else:
            return 'Verification failed', 403

    elif request.method == 'POST':
        # Process webhook messages
        data = request.get_json()
        ai_workforce = load_data(AI_WORKFORCE_FILE)

        if ai_id not in ai_workforce:
            return jsonify({'error': 'AI not found'}), 404

        ai_config = ai_workforce[ai_id]
        user_id = ai_config['user_id']

        for entry in data.get('entry', []):
            for change in entry.get('changes', []):
                process_facebook_change(change, user_id, ai_id)

        return jsonify({'status': 'ok'})

@app.route('/facebook/post/<ai_id>', methods=['POST'])
@login_required
def facebook_post(ai_id):
    try:
        message_prompt = request.form['message']
        ai_response = generate_ai_response(
            f"Create a Facebook post about: {message_prompt}",
            session['user_id'],
            ai_id
        )

        access_token_data = get_facebook_token(session['user_id'], ai_id)
        if not access_token_data:
            flash('Facebook not configured. Please setup your access token first.', 'error')
            return redirect(url_for('dashboard'))

        access_token = access_token_data['access_token']
        page_id = access_token_data.get('page_id')

        if not page_id:
            # Try to get page ID from token if not provided
            flash('Page ID is required for posting', 'error')
            return redirect(url_for('dashboard'))

        result = post_to_facebook_page(page_id, access_token, ai_response)

        if result:
            flash('Posted to Facebook successfully!', 'success')
            save_chat_history(
                session['user_id'],
                ai_id,
                'facebook_post',
                page_id,
                message_prompt,
                ai_response,
                "outgoing"
            )
        else:
            flash('Failed to post to Facebook.', 'error')

        return redirect(url_for('dashboard'))

    except Exception as e:
        flash(f'Error creating Facebook post: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/facebook/analyze/<ai_id>', methods=['POST'])
@login_required
def facebook_analyze(ai_id):
    try:
        access_token_data = get_facebook_token(session['user_id'], ai_id)
        if not access_token_data:
            flash('Facebook not connected. Please setup your access token first.', 'error')
            return redirect(url_for('dashboard'))

        access_token = access_token_data['access_token']
        page_id = access_token_data.get('page_id')

        if not page_id:
            flash('Page ID is required for analysis', 'error')
            return redirect(url_for('dashboard'))

        # Get analysis period from form
        days = int(request.form.get('analysis_period', 7))

        # Perform comprehensive analysis
        analysis = analyze_content_performance(access_token, page_id, days)

        if 'error' in analysis:
            flash(f'Analysis failed: {analysis["error"]}', 'error')
            return redirect(url_for('dashboard'))

        # Generate AI recommendations
        ai_workforce = load_data(AI_WORKFORCE_FILE)
        ai_config = ai_workforce.get(ai_id, {})
        business_context = ai_config.get('business_info', '')

        analysis_prompt = f"""Analyze these Facebook insights and provide specific recommendations:
        {json.dumps(analysis, indent=2)}

        Business Context: {business_context}

        Provide specific recommendations for:
        1. Best posting times
        2. Content types that perform well
        3. Engagement strategies
        4. Content ideas for future posts"""

        ai_analysis = generate_ai_response(analysis_prompt, session['user_id'], ai_id)

        flash('Facebook analysis completed successfully!', 'success')
        save_chat_history(
            session['user_id'],
            ai_id,
            'facebook_analysis',
            page_id,
            f"Requested page analysis for {days} days",
            ai_analysis,
            "outgoing"
        )

        # Store analysis results for later use
        analysis['ai_recommendations'] = ai_analysis
        analysis['analyzed_at'] = datetime.now().isoformat()

        # Save to content strategy
        strategy_data = {
            'performance_data': analysis,
            'last_analyzed': datetime.now().isoformat()
        }
        save_content_strategy(session['user_id'], ai_id, strategy_data)

        return render_template_string(ANALYSIS_RESULTS_TEMPLATE,
                                   analysis=analysis,
                                   ai_analysis=ai_analysis,
                                   ai_id=ai_id,
                                   business_context=business_context)

    except Exception as e:
        flash(f'Error analyzing Facebook: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

def process_generic_webhook(update, user_id, ai_id):
    try:
        platform = request.args.get('platform', 'generic')
        sender_id = request.args.get('sender_id', 'unknown')
        message_text = update.get('text', '') or update.get('message', '') or str(update)

        if not message_text:
            return jsonify({'error': 'No message content'}), 400

        chat_history = get_user_chat_history(user_id, ai_id, platform)
        conversation_context = []

        for conv_id, conv_data in chat_history.items():
            if sender_id in conv_id:
                for msg in conv_data['messages']:
                    if msg['direction'] == 'incoming':
                        conversation_context.append({"role": "user", "content": msg['message']})
                    else:
                        conversation_context.append({"role": "assistant", "content": msg['response']})

        ai_response = generate_ai_response(message_text, user_id, ai_id, conversation_context)
        save_chat_history(user_id, ai_id, platform, sender_id, message_text, ai_response, "incoming")

        return jsonify({
            'status': 'success',
            'response': ai_response,
            'platform': platform,
            'sender_id': sender_id
        })

    except Exception as e:
        print(f"Error processing generic webhook: {e}")
        return jsonify({'error': str(e)}), 500

# Database routes
@app.route('/database_upload')
@login_required
def database_upload():
    user_databases = load_data(USER_DATABASES_FILE)
    user_db_list = [db for db in user_databases.values() if db['user_id'] == session['user_id']]
    return render_template_string(DATABASE_UPLOAD_TEMPLATE, databases=user_db_list)

@app.route('/upload_database', methods=['POST'])
@login_required
def upload_database():
    try:
        if 'database_file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('database_upload'))

        file = request.files['database_file']
        db_name = request.form.get('db_name', '').strip()
        db_description = request.form.get('db_description', '').strip()

        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('database_upload'))

        if not db_name:
            flash('Database name is required', 'error')
            return redirect(url_for('database_upload'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_extension = filename.rsplit('.', 1)[1].lower()

            # Read and validate the file content
            file_content = file.read().decode('utf-8')

            try:
                if file_extension == 'json':
                    data = json.loads(file_content)
                elif file_extension == 'csv':
                    csv_content = file_content.splitlines()
                    if not csv_content:
                        flash('CSV file is empty', 'error')
                        return redirect(url_for('database_upload'))

                    csv_reader = csv.DictReader(csv_content)
                    data = list(csv_reader)

                    if not data:
                        flash('CSV file contains no data', 'error')
                        return redirect(url_for('database_upload'))
                else:
                    flash('Unsupported file format', 'error')
                    return redirect(url_for('database_upload'))

                # Save database metadata
                user_databases = load_data(USER_DATABASES_FILE)
                db_id = str(uuid.uuid4())

                user_databases[db_id] = {
                    'id': db_id,
                    'user_id': session['user_id'],
                    'name': db_name,
                    'description': db_description,
                    'filename': filename,
                    'file_type': file_extension,
                    'record_count': len(data) if isinstance(data, list) else 1,
                    'uploaded_at': datetime.now().isoformat(),
                    'data': data
                }

                save_data(user_databases, USER_DATABASES_FILE)
                flash('Database uploaded successfully!', 'success')

            except (json.JSONDecodeError, csv.Error) as e:
                flash(f'Invalid file format: {str(e)}', 'error')
                return redirect(url_for('database_upload'))

        else:
            flash('Only JSON and CSV files are allowed', 'error')

        return redirect(url_for('database_upload'))

    except Exception as e:
        flash(f'Error uploading database: {str(e)}', 'error')
        return redirect(url_for('database_upload'))

@app.route('/delete_database/<db_id>')
@login_required
def delete_database(db_id):
    user_databases = load_data(USER_DATABASES_FILE)
    if db_id in user_databases and user_databases[db_id]['user_id'] == session['user_id']:
        del user_databases[db_id]
        save_data(user_databases, USER_DATABASES_FILE)
        flash('Database deleted successfully!', 'success')
    else:
        flash('Database not found', 'error')
    return redirect(url_for('database_upload'))

# Lookup worker
@app.route('/lookup_worker')
def lookup_worker():
    key = request.args.get('key')
    lookup_type = request.args.get('type', 'auto_detect')
    user_id = request.args.get('user_id')
    ai_id = request.args.get('ai_id')
    context = request.args.get('context', '')

    if lookup_type == 'auto_detect':
        lookup_type = detect_lookup_type(key, context)

    result = {
        'status': 'success',
        'key': key,
        'type': lookup_type,
        'timestamp': datetime.now().isoformat(),
        'data': {}
    }

    if lookup_type == 'user_info':
        result['data'] = lookup_user_info(key or user_id)
    elif lookup_type == 'user_database':
        if key:
            search_results = search_user_databases(user_id, key)
            result['data'] = {
                'search_query': key,
                'results_found': len(search_results),
                'results': search_results
            }
        else:
            result['data'] = {'error': 'No search query provided'}
    else:
        result['data'] = generic_lookup(key, context)

    return jsonify(result)

def detect_lookup_type(key, context):
    context_lower = context.lower() if context else ""
    if any(word in context_lower for word in ['database', 'record', 'data', 'lookup', 'search', 'find']):
        return 'user_database'
    return 'generic'

def lookup_user_info(user_id):
    users = load_data(USERS_FILE)
    if user_id in users:
        return {
            'user_exists': True,
            'email': users[user_id]['email'],
            'business_name': users[user_id]['business_name'],
            'user_since': users[user_id].get('created_at', 'Unknown')
        }
    return {'user_exists': False}

def generic_lookup(key, context):
    return {
        'message': f'Lookup performed for: {key}',
        'context': context,
        'result': f'Information about {key}'
    }

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# HTML Templates with mobile-responsive design
LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - AI Workforce</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .login-container {
            max-width: 400px;
            width: 100%;
        }
        .card {
            border-radius: 1rem;
            box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
            border: none;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="card">
            <div class="card-body p-4">
                <h2 class="card-title text-center mb-4 text-dark">Login</h2>
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                                {{ message }}
                                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                            </div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
                <form method="POST">
                    <div class="mb-3">
                        <label for="email" class="form-label">Email</label>
                        <input type="email" class="form-control" id="email" name="email" required>
                    </div>
                    <div class="mb-3">
                        <label for="password" class="form-label">Password</label>
                        <input type="password" class="form-control" id="password" name="password" required>
                    </div>
                    <button type="submit" class="btn btn-primary w-100 py-2">Login</button>
                </form>
                <hr>
                <div class="text-center">
                    <p class="mb-0">Don't have an account? <a href="{{ url_for('signup') }}" class="text-decoration-none">Sign up</a></p>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

SIGNUP_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Up - AI Workforce</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .signup-container {
            max-width: 500px;
            width: 100%;
        }
        .card {
            border-radius: 1rem;
            box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
            border: none;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
        }
    </style>
</head>
<body>
    <div class="signup-container">
        <div class="card">
            <div class="card-body p-4">
                <h2 class="card-title text-center mb-4 text-dark">Create Account</h2>
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                                {{ message }}
                                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                            </div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
                <form method="POST">
                    <div class="mb-3">
                        <label for="business_name" class="form-label">Business Name</label>
                        <input type="text" class="form-control" id="business_name" name="business_name" required>
                    </div>
                    <div class="mb-3">
                        <label for="email" class="form-label">Email</label>
                        <input type="email" class="form-control" id="email" name="email" required>
                    </div>
                    <div class="mb-3">
                        <label for="password" class="form-label">Password</label>
                        <input type="password" class="form-control" id="password" name="password" required>
                    </div>
                    <div class="mb-3">
                        <label for="confirm_password" class="form-label">Confirm Password</label>
                        <input type="password" class="form-control" id="confirm_password" name="confirm_password" required>
                    </div>
                    <button type="submit" class="btn btn-primary w-100 py-2">Sign Up</button>
                </form>
                <hr>
                <div class="text-center">
                    <p class="mb-0">Already have an account? <a href="{{ url_for('login') }}" class="text-decoration-none">Login</a></p>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - AI Workforce</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    <style>
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --sidebar-width: 250px;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
        }

        .sidebar {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            height: 100vh;
            position: fixed;
            left: 0;
            top: 0;
            width: var(--sidebar-width);
            padding: 0;
            transition: all 0.3s ease;
            z-index: 1000;
        }

        .sidebar-brand {
            padding: 1.5rem 1rem;
            color: white;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .sidebar .list-group-item {
            background: transparent;
            border: none;
            color: rgba(255,255,255,0.8);
            border-radius: 0;
            padding: 1rem 1.5rem;
            transition: all 0.3s ease;
        }

        .sidebar .list-group-item:hover,
        .sidebar .list-group-item.active {
            background: rgba(255,255,255,0.1);
            color: white;
        }

        .main-content {
            margin-left: var(--sidebar-width);
            padding: 20px;
            transition: all 0.3s ease;
        }

        .ai-card {
            transition: transform 0.2s, box-shadow 0.2s;
            border: none;
            border-radius: 1rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }

        .ai-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
        }

        .stats-card {
            background: white;
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }

        @media (max-width: 768px) {
            .sidebar {
                width: 100%;
                left: -100%;
            }

            .sidebar.show {
                left: 0;
            }

            .main-content {
                margin-left: 0;
                padding: 15px;
            }

            .sidebar-toggle {
                display: block !important;
                position: fixed;
                top: 15px;
                left: 15px;
                z-index: 1001;
            }
        }

        .badge-online {
            background: linear-gradient(135deg, #28a745, #20c997);
        }

        .badge-offline {
            background: linear-gradient(135deg, #dc3545, #fd7e14);
        }

        .webhook-url {
            font-size: 0.8rem;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 0.375rem;
            padding: 0.375rem 0.75rem;
        }

        .form-control-sm {
            font-size: 0.75rem;
        }

        .btn-sm {
            padding: 0.25rem 0.5rem;
            font-size: 0.75rem;
        }
    </style>
</head>
<body>
    <!-- Mobile Sidebar Toggle -->
    <button class="btn btn-primary sidebar-toggle d-none" type="button" onclick="toggleSidebar()">
        <i class="bi bi-list"></i>
    </button>

    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-3 col-lg-2 sidebar p-0" id="sidebar">
                <div class="sidebar-brand">
                    <h4 class="mb-0">AI Workforce</h4>
                    <small class="text-white-50">Business Hub</small>
                </div>
                <div class="list-group mt-3">
                    <a href="{{ url_for('dashboard') }}" class="list-group-item list-group-item-action active">
                        <i class="bi bi-speedometer2 me-2"></i>Dashboard
                    </a>
                    <a href="{{ url_for('api_settings') }}" class="list-group-item list-group-item-action">
                        <i class="bi bi-key me-2"></i>API Settings
                    </a>
                    <a href="{{ url_for('database_upload') }}" class="list-group-item list-group-item-action">
                        <i class="bi bi-database me-2"></i>My Databases
                    </a>
                    <a href="{{ url_for('chat_history') }}" class="list-group-item list-group-item-action">
                        <i class="bi bi-chat-dots me-2"></i>Chat History
                    </a>
                    <a href="{{ url_for('logout') }}" class="list-group-item list-group-item-action">
                        <i class="bi bi-box-arrow-right me-2"></i>Logout
                    </a>
                </div>
            </div>

            <!-- Main content -->
            <div class="col-md-9 col-lg-10 main-content">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <div>
                        <h2 class="mb-1">Welcome, {{ business_name }}</h2>
                        <span class="text-muted">{{ session.get('user_email', 'User') }}</span>
                    </div>
                    <div class="d-none d-md-block">
                        <span class="badge bg-primary">Active AI: {{ user_ai|length }}</span>
                    </div>
                </div>

                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                                {{ message }}
                                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                            </div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}

                <!-- Quick Stats -->
                <div class="row mb-4">
                    <div class="col-6 col-md-3 mb-3">
                        <div class="stats-card text-center">
                            <i class="bi bi-robot fs-1 text-primary mb-2"></i>
                            <h3>{{ user_ai|length }}</h3>
                            <p class="text-muted mb-0">AI Agents</p>
                        </div>
                    </div>
                    <div class="col-6 col-md-3 mb-3">
                        <div class="stats-card text-center">
                            <i class="bi bi-chat-dots fs-1 text-success mb-2"></i>
                            <h3>{{ user_chat_count }}</h3>
                            <p class="text-muted mb-0">Conversations</p>
                        </div>
                    </div>
                    <div class="col-6 col-md-3 mb-3">
                        <div class="stats-card text-center">
                            <i class="bi bi-database fs-1 text-info mb-2"></i>
                            <h3>{{ user_db_count }}</h3>
                            <p class="text-muted mb-0">Databases</p>
                        </div>
                    </div>
                    <div class="col-6 col-md-3 mb-3">
                        <div class="stats-card text-center">
                            <i class="bi bi-activity fs-1 text-warning mb-2"></i>
                            <h3>{{ active_ai_count }}</h3>
                            <p class="text-muted mb-0">Active</p>
                        </div>
                    </div>
                </div>

                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header bg-white">
                                <h5 class="card-title mb-0">Your AI Workforce</h5>
                            </div>
                            <div class="card-body">
                                {% if user_ai %}
                                    <div class="row">
                                        {% for ai in user_ai %}
                                            <div class="col-12 col-md-6 col-lg-4 mb-3">
                                                <div class="card ai-card h-100">
                                                    <div class="card-body">
                                                        <div class="d-flex justify-content-between align-items-start mb-2">
                                                            <h5 class="card-title">{{ ai.name }}</h5>
                                                            <span class="badge {% if ai.status == 'active' %}badge-online{% else %}badge-offline{% endif %}">
                                                                {{ ai.status }}
                                                            </span>
                                                        </div>
                                                        <h6 class="card-subtitle mb-2 text-muted">
                                                            <i class="bi bi-{{ ai.ai_id }} me-1"></i>{{ ai.service_name }}
                                                        </h6>
                                                        <p class="card-text small">{{ ai.description }}</p>

                                                        {% if ai.ai_id == 'telegram' %}
                                                        <div class="mb-2">
                                                            <small class="text-muted d-block mb-1">Webhook URL:</small>
                                                            <input type="text" class="form-control form-control-sm webhook-url"
                                                                   value="{{ ai.webhook_url }}"
                                                                   readonly onclick="this.select()">
                                                        </div>
                                                        <form method="POST" action="{{ url_for('setup_telegram_bot') }}">
                                                            <input type="hidden" name="ai_id" value="{{ ai.id }}">
                                                            <div class="input-group input-group-sm mb-2">
                                                                <input type="text" class="form-control" placeholder="Telegram Bot Token"
                                                                       name="bot_token" required>
                                                                <button class="btn btn-outline-primary" type="submit">Set Webhook</button>
                                                            </div>
                                                        </form>
                                                        {% endif %}

                                                        {% if ai.ai_id == 'whatsapp' %}
                                                        <div class="mb-2">
                                                            <small class="text-muted d-block mb-1">Webhook URL:</small>
                                                            <input type="text" class="form-control form-control-sm webhook-url"
                                                                   value="{{ ai.webhook_url }}"
                                                                   readonly onclick="this.select()">
                                                        </div>
                                                        <form method="POST" action="{{ url_for('setup_whatsapp_bot') }}">
                                                            <input type="hidden" name="ai_id" value="{{ ai.id }}">
                                                            <div class="mb-2">
                                                                <select class="form-select form-select-sm" name="provider" required>
                                                                    <option value="">Select Provider</option>
                                                                    <option value="twilio">Twilio</option>
                                                                    <option value="wati">WATI.io</option>
                                                                    <option value="generic">Generic WhatsApp API</option>
                                                                </select>
                                                            </div>
                                                            <div class="mb-2">
                                                                <input type="text" class="form-control form-control-sm"
                                                                       placeholder="Phone Number" name="phone_number" required>
                                                            </div>
                                                            <div class="mb-2">
                                                                <input type="text" class="form-control form-control-sm"
                                                                       placeholder="Account SID/API Key" name="api_key" required>
                                                            </div>
                                                            <div class="mb-2">
                                                                <input type="text" class="form-control form-control-sm"
                                                                       placeholder="Auth Token (if applicable)" name="auth_token">
                                                            </div>
                                                            <div class="mb-2">
                                                                <input type="text" class="form-control form-control-sm"
                                                                       placeholder="WhatsApp Number/Phone Number ID" name="whatsapp_number">
                                                            </div>
                                                            <button class="btn btn-outline-primary btn-sm w-100" type="submit">Configure WhatsApp</button>
                                                        </form>
                                                        {% endif %}

                                                        {% if ai.ai_id == 'facebook' %}
<div class="mb-2">
    <small class="text-muted d-block mb-1">Webhook URL:</small>
    <input type="text" class="form-control form-control-sm webhook-url"
           value="{{ request.host_url }}webhook/facebook/{{ ai.id }}"
           readonly onclick="this.select()">
</div>
<div class="d-flex flex-column gap-2">
    <a href="{{ url_for('facebook_setup', ai_id=ai.id) }}" class="btn btn-outline-primary btn-sm">
        {% if ai.has_facebook_token %}Edit{% else %}Setup{% endif %} Facebook Token
    </a>

    {% if ai.has_facebook_token %}
    <div class="mt-2">
        <button class="btn btn-outline-info btn-sm w-100" type="button" data-bs-toggle="collapse"
                data-bs-target="#facebookActions{{ ai.id }}">
            Facebook Actions
        </button>

        <div class="collapse mt-2" id="facebookActions{{ ai.id }}">
            <div class="card card-body">
                <form method="POST" action="{{ url_for('facebook_post', ai_id=ai.id) }}">
                    <div class="mb-2">
                        <label class="form-label">Create Post</label>
                        <textarea class="form-control form-control-sm" name="message"
                                  placeholder="What would you like to post?" rows="3"></textarea>
                    </div>
                    <button type="submit" class="btn btn-primary btn-sm w-100">Create with AI</button>
                </form>

                <hr>

                <form method="POST" action="{{ url_for('facebook_analyze', ai_id=ai.id) }}">
                    <button type="submit" class="btn btn-info btn-sm w-100">
                        Analyze Page Performance
                    </button>
                </form>
            </div>
        </div>
    </div>
    {% endif %}
</div>
{% endif %}

                                                        <div class="d-flex justify-content-between align-items-center mt-3 pt-2 border-top">
                                                            <small class="text-muted">{{ ai.created_at[:10] if ai.created_at else 'N/A' }}</small>
                                                            <a href="{{ url_for('delete_ai', ai_id=ai.id) }}" class="text-danger"
                                                               onclick="return confirm('Are you sure you want to delete this AI?')">
                                                                <i class="bi bi-trash"></i>
                                                            </a>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        {% endfor %}
                                    </div>
                                {% else %}
                                    <div class="text-center py-5">
                                        <i class="bi bi-robot fs-1 text-muted mb-3"></i>
                                        <p class="text-muted">You haven't added any AI workforce yet.</p>
                                        <a href="#add-ai-section" class="btn btn-primary">Add Your First AI</a>
                                    </div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="row" id="add-ai-section">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header bg-white">
                                <h5 class="card-title mb-0">Add AI to Your Workforce</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    {% for ai in available_ai %}
                                        <div class="col-12 col-sm-6 col-md-4 col-lg-3 mb-3">
                                            <div class="card ai-card h-100">
                                                <div class="card-body text-center">
                                                    <div class="mb-3" style="font-size: 2.5rem;">
                                                        <i class="bi bi-{{ ai.id }} text-primary"></i>
                                                    </div>
                                                    <h5 class="card-title">{{ ai.name }}</h5>
                                                    <p class="card-text small text-muted">{{ ai.description }}</p>
                                                    <a href="{{ url_for('add_ai', ai_id=ai.id) }}" class="btn btn-primary btn-sm">Add to My Workforce</a>
                                                </div>
                                            </div>
                                        </div>
                                    {% endfor %}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            sidebar.classList.toggle('show');
        }

        // Auto-hide sidebar on mobile when clicking outside
        document.addEventListener('click', function(event) {
            const sidebar = document.getElementById('sidebar');
            const toggleBtn = document.querySelector('.sidebar-toggle');

            if (window.innerWidth <= 768 &&
                sidebar.classList.contains('show') &&
                !sidebar.contains(event.target) &&
                event.target !== toggleBtn &&
                !toggleBtn.contains(event.target)) {
                sidebar.classList.remove('show');
            }
        });

        // Show sidebar toggle on mobile
        function checkMobile() {
            const toggleBtn = document.querySelector('.sidebar-toggle');
            if (window.innerWidth <= 768) {
                toggleBtn.classList.remove('d-none');
            } else {
                toggleBtn.classList.add('d-none');
                const sidebar = document.getElementById('sidebar');
                if (sidebar) sidebar.classList.remove('show');
            }
        }

        // Check on load and resize
        window.addEventListener('load', checkMobile);
        window.addEventListener('resize', checkMobile);
    </script>
</body>
</html>
'''

      # Add these template constants after the DASHBOARD_TEMPLATE

ADD_AI_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Add {{ ai_service.name }} - AI Workforce</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .card {
            border-radius: 1rem;
            box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
            border: none;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-8 col-lg-6">
                <div class="card">
                    <div class="card-body p-4">
                        <div class="text-center mb-4">
                            <i class="bi bi-{{ ai_service.id }} fs-1 text-primary mb-3"></i>
                            <h2 class="card-title">Add {{ ai_service.name }}</h2>
                            <p class="text-muted">{{ ai_service.description }}</p>
                        </div>

                        {% with messages = get_flashed_messages(with_categories=true) %}
                            {% if messages %}
                                {% for category, message in messages %}
                                    <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                                        {{ message }}
                                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                                    </div>
                                {% endfor %}
                            {% endif %}
                        {% endwith %}

                        <form method="POST" action="{{ url_for('save_ai') }}">
                            <input type="hidden" name="ai_id" value="{{ ai_service.id }}">

                            <div class="mb-3">
                                <label for="ai_name" class="form-label">AI Name</label>
                                <input type="text" class="form-control" id="ai_name" name="ai_name"
                                       placeholder="e.g., My Customer Support AI" required>
                            </div>

                            <div class="mb-3">
                                <label for="description" class="form-label">Description</label>
                                <textarea class="form-control" id="description" name="description"
                                          rows="2" placeholder="Brief description of this AI's purpose"></textarea>
                            </div>

                            <div class="mb-4">
                                <label for="business_info" class="form-label">Business Context</label>
                                <textarea class="form-control" id="business_info" name="business_info"
                                          rows="4" placeholder="Provide information about your business, products/services, and common customer queries. This helps the AI understand your business context." required></textarea>
                                <div class="form-text">This information will be used to train the AI about your business.</div>
                            </div>

                            <div class="d-grid gap-2">
                                <button type="submit" class="btn btn-primary py-2">Create AI Assistant</button>
                                <a href="{{ url_for('dashboard') }}" class="btn btn-outline-secondary py-2">Cancel</a>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

API_SETTINGS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Settings - AI Workforce</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    <style>
        body {
            background: #f8f9fa;
            padding: 20px;
        }
        .card {
            border-radius: 1rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
            border: none;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-10">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h2 class="mb-0">API Settings</h2>
                    <a href="{{ url_for('dashboard') }}" class="btn btn-outline-secondary">
                        <i class="bi bi-arrow-left"></i> Back to Dashboard
                    </a>
                </div>

                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                                {{ message }}
                                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                            </div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}

                <div class="card mb-4">
                    <div class="card-header bg-white">
                        <h5 class="card-title mb-0">Add API Key</h5>
                    </div>
                    <div class="card-body">
                        <form method="POST" action="{{ url_for('save_api_key') }}">
                            <div class="row">
                                <div class="col-md-6 mb-3">
                                    <label for="service_name" class="form-label">Service</label>
                                    <select class="form-select" id="service_name" name="service_name" required>
                                        <option value="">Select Service</option>
                                        <option value="OpenRouter">OpenRouter API</option>
                                        <option value="OpenAI">OpenAI API</option>
                                        <option value="Twilio">Twilio</option>
                                        <option value="WhatsApp">WhatsApp Business API</option>
                                        <option value="Telegram">Telegram Bot API</option>
                                    </select>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="api_key" class="form-label">API Key</label>
                                    <input type="password" class="form-control" id="api_key" name="api_key"
                                           placeholder="Enter your API key" required>
                                </div>
                            </div>
                            <button type="submit" class="btn btn-primary">Save API Key</button>
                        </form>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header bg-white">
                        <h5 class="card-title mb-0">Your API Keys</h5>
                    </div>
                    <div class="card-body">
                        {% if api_keys %}
                            <div class="table-responsive">
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>Service</th>
                                            <th>API Key</th>
                                            <th>Created</th>
                                            <th>Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for key in api_keys %}
                                            <tr>
                                                <td>{{ key.service_name }}</td>
                                                <td>
                                                    <code>{{ key.api_key[:4] }}••••{{ key.api_key[-4:] if key.api_key|length > 8 else '••••' }}</code>
                                                </td>
                                                <td>{{ key.created_at|string|truncate(10, True, '') }}</td>
                                                <td>
                                                    <a href="{{ url_for('delete_api_key', key_id=key.id) }}"
                                                       class="btn btn-sm btn-outline-danger"
                                                       onclick="return confirm('Are you sure you want to delete this API key?')">
                                                        <i class="bi bi-trash"></i>
                                                    </a>
                                                </td>
                                            </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        {% else %}
                            <div class="text-center py-4">
                                <i class="bi bi-key fs-1 text-muted mb-3"></i>
                                <p class="text-muted">No API keys configured yet.</p>
                            </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

CHAT_HISTORY_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat History - AI Workforce</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    <style>
        body {
            background: #f8f9fa;
            padding: 20px;
        }
        .card {
            border-radius: 1rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
            border: none;
        }
        .chat-message {
            border-radius: 1rem;
            padding: 1rem;
            margin-bottom: 1rem;
            max-width: 80%;
        }
        .incoming {
            background: #e3f2fd;
            margin-right: auto;
        }
        .outgoing {
            background: #f1f8e9;
            margin-left: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-12">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h2 class="mb-0">Chat History</h2>
                    <a href="{{ url_for('dashboard') }}" class="btn btn-outline-secondary">
                        <i class="bi bi-arrow-left"></i> Back to Dashboard
                    </a>
                </div>

                <div class="card mb-4">
                    <div class="card-header bg-white">
                        <h5 class="card-title mb-0">Filters</h5>
                    </div>
                    <div class="card-body">
                        <form method="GET" class="row g-3">
                            <div class="col-md-4">
                                <label for="ai_id" class="form-label">AI Assistant</label>
                                <select class="form-select" id="ai_id" name="ai_id">
                                    <option value="">All Assistants</option>
                                    {% for ai_id, ai_data in ai_workforce.items() %}
                                        {% if ai_data.user_id == session.user_id %}
                                            <option value="{{ ai_id }}" {% if request.args.get('ai_id') == ai_id %}selected{% endif %}>
                                                {{ ai_data.name }}
                                            </option>
                                        {% endif %}
                                    {% endfor %}
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label for="platform" class="form-label">Platform</label>
                                <select class="form-select" id="platform" name="platform">
                                    <option value="">All Platforms</option>
                                    <option value="telegram" {% if request.args.get('platform') == 'telegram' %}selected{% endif %}>Telegram</option>
                                    <option value="whatsapp" {% if request.args.get('platform') == 'whatsapp' %}selected{% endif %}>WhatsApp</option>
                                    <option value="facebook" {% if request.args.get('platform') == 'facebook' %}selected{% endif %}>Facebook</option>
                                    <option value="generic" {% if request.args.get('platform') == 'generic' %}selected{% endif %}>Generic</option>
                                </select>
                            </div>
                            <div class="col-md-4 d-flex align-items-end">
                                <button type="submit" class="btn btn-primary me-2">Apply Filters</button>
                                <a href="{{ url_for('chat_history') }}" class="btn btn-outline-secondary">Clear</a>
                            </div>
                        </form>
                    </div>
                </div>

                {% if chats %}
                    <div class="card">
                        <div class="card-header bg-white">
                            <h5 class="card-title mb-0">Conversations ({{ chats|length }})</h5>
                        </div>
                        <div class="card-body">
                            <div class="accordion" id="chatAccordion">
                                {% for conv_id, conv_data in chats.items() %}
                                    <div class="accordion-item">
                                        <h2 class="accordion-header" id="heading{{ loop.index }}">
                                            <button class="accordion-button collapsed" type="button"
                                                    data-bs-toggle="collapse" data-bs-target="#collapse{{ loop.index }}"
                                                    aria-expanded="false" aria-controls="collapse{{ loop.index }}">
                                                <div class="d-flex justify-content-between w-100 me-3">
                                                    <div>
                                                        <strong>{{ conv_data.platform|title }}</strong> •
                                                        {{ ai_workforce[conv_data.ai_id].name if conv_data.ai_id in ai_workforce else 'Unknown AI' }}
                                                    </div>
                                                    <div class="text-muted">
                                                        {{ conv_data.updated_at|string|truncate(10, True, '') }}
                                                    </div>
                                                </div>
                                            </button>
                                        </h2>
                                        <div id="collapse{{ loop.index }}" class="accordion-collapse collapse"
                                             aria-labelledby="heading{{ loop.index }}" data-bs-parent="#chatAccordion">
                                            <div class="accordion-body">
                                                <div class="d-flex justify-content-between align-items-center mb-3">
                                                    <small class="text-muted">
                                                        Conversation ID: {{ conv_id }} •
                                                        Started: {{ conv_data.created_at|string|truncate(16, True, '') }}
                                                    </small>
                                                </div>

                                                <div class="chat-container">
                                                    {% for message in conv_data.messages %}
                                                        <div class="chat-message {% if message.direction == 'incoming' %}incoming{% else %}outgoing{% endif %}">
                                                            <div class="d-flex justify-content-between align-items-start mb-1">
                                                                <strong>{% if message.direction == 'incoming' %}Customer{% else %}AI{% endif %}:</strong>
                                                                <small class="text-muted">
                                                                    {{ message.timestamp|string|truncate(16, True, '') }}
                                                                </small>
                                                            </div>
                                                            <p class="mb-1">{{ message.message }}</p>
                                                            {% if message.response %}
                                                                <div class="mt-2 p-2 bg-white rounded">
                                                                    <strong>Response:</strong>
                                                                    <p class="mb-0">{{ message.response }}</p>
                                                                </div>
                                                            {% endif %}
                                                        </div>
                                                    {% endfor %}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                {% endfor %}
                            </div>
                        </div>
                    </div>
                {% else %}
                    <div class="card">
                        <div class="card-body text-center py-5">
                            <i class="bi bi-chat-dots fs-1 text-muted mb-3"></i>
                            <p class="text-muted">No chat history found.</p>
                            <p class="text-muted small">Start conversations with your AI assistants to see history here.</p>
                        </div>
                    </div>
                {% endif %}
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

DATABASE_UPLOAD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Database Upload - AI Workforce</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    <style>
        body {
            background: #f8f9fa;
            padding: 20px;
        }
        .card {
            border-radius: 1rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
            border: none;
        }
        .database-card {
            transition: transform 0.2s;
        }
        .database-card:hover {
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-12">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h2 class="mb-0">My Databases</h2>
                    <a href="{{ url_for('dashboard') }}" class="btn btn-outline-secondary">
                        <i class="bi bi-arrow-left"></i> Back to Dashboard
                    </a>
                </div>

                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                                {{ message }}
                                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                            </div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}

                <div class="row">
                    <div class="col-md-6 mb-4">
                        <div class="card h-100">
                            <div class="card-header bg-white">
                                <h5 class="card-title mb-0">Upload New Database</h5>
                            </div>
                            <div class="card-body">
                                <form method="POST" action="{{ url_for('upload_database') }}" enctype="multipart/form-data">
                                    <div class="mb-3">
                                        <label for="db_name" class="form-label">Database Name</label>
                                        <input type="text" class="form-control" id="db_name" name="db_name" required>
                                    </div>

                                    <div class="mb-3">
                                        <label for="db_description" class="form-label">Description</label>
                                        <textarea class="form-control" id="db_description" name="db_description"
                                                  rows="2" placeholder="What type of data does this contain?"></textarea>
                                    </div>

                                    <div class="mb-3">
                                        <label for="database_file" class="form-label">Database File</label>
                                        <input type="file" class="form-control" id="database_file" name="database_file"
                                               accept=".json,.csv" required>
                                        <div class="form-text">Supported formats: JSON, CSV (Max 16MB)</div>
                                    </div>

                                    <button type="submit" class="btn btn-primary w-100">
                                        <i class="bi bi-upload"></i> Upload Database
                                    </button>
                                </form>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-white">
                                <h5 class="card-title mb-0">Your Databases ({{ databases|length }})</h5>
                            </div>
                            <div class="card-body">
                                {% if databases %}
                                    <div class="row">
                                        {% for db in databases %}
                                            <div class="col-12 mb-3">
                                                <div class="card database-card h-100">
                                                    <div class="card-body">
                                                        <div class="d-flex justify-content-between align-items-start mb-2">
                                                            <h6 class="card-title mb-0">{{ db.name }}</h6>
                                                            <span class="badge bg-secondary">{{ db.file_type|upper }}</span>
                                                        </div>
                                                        <p class="card-text small text-muted mb-2">
                                                            {{ db.description or 'No description' }}
                                                        </p>
                                                        <div class="d-flex justify-content-between align-items-center">
                                                            <small class="text-muted">
                                                                {{ db.record_count }} records •
                                                                {{ db.uploaded_at|string|truncate(10, True, '') }}
                                                            </small>
                                                            <a href="{{ url_for('delete_database', db_id=db.id) }}"
                                                               class="btn btn-sm btn-outline-danger"
                                                               onclick="return confirm('Are you sure you want to delete this database?')">
                                                                <i class="bi bi-trash"></i>
                                                            </a>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        {% endfor %}
                                    </div>
                                {% else %}
                                    <div class="text-center py-4">
                                        <i class="bi bi-database fs-1 text-muted mb-3"></i>
                                        <p class="text-muted">No databases uploaded yet.</p>
                                        <p class="text-muted small">Upload JSON or CSV files to enable AI data lookup.</p>
                                    </div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

# Add this JavaScript function to handle sidebar toggling
SIDEBAR_JS = '''
<script>
    function toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.toggle('show');
    }

    // Auto-hide sidebar on mobile when clicking outside
    document.addEventListener('click', function(event) {
        const sidebar = document.getElementById('sidebar');
        const toggleBtn = document.querySelector('.sidebar-toggle');

        if (window.innerWidth <= 768 &&
            sidebar.classList.contains('show') &&
            !sidebar.contains(event.target) &&
            event.target !== toggleBtn &&
            !toggleBtn.contains(event.target)) {
            sidebar.classList.remove('show');
        }
    });

    // Initialize tooltips
    document.addEventListener('DOMContentLoaded', function() {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        const tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    });
</script>
'''

FACEBOOK_SETUP_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Facebook Setup - AI Workforce</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: #f8f9fa;
            padding: 20px;
        }
        .card {
            border-radius: 1rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-8 col-lg-6">
                <div class="card">
                    <div class="card-body p-4">
                        <h2 class="card-title text-center mb-4">Facebook Setup</h2>

                        {% with messages = get_flashed_messages(with_categories=true) %}
                            {% if messages %}
                                {% for category, message in messages %}
                                    <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                                        {{ message }}
                                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                                    </div>
                                {% endfor %}
                            {% endif %}
                        {% endwith %}

                        <form method="POST">
                            <div class="mb-3">
                                <label for="access_token" class="form-label">Facebook Page Access Token</label>
                                <textarea class="form-control" id="access_token" name="access_token"
                                          rows="3" placeholder="Paste your Facebook Page Access Token here"
                                          required>{{ current_token.access_token if current_token else '' }}</textarea>
                                <div class="form-text">
                                    Get this from your Facebook Developer account or Page Settings
                                </div>
                            </div>

                            <div class="mb-3">
                                <label for="page_id" class="form-label">Page ID (Optional)</label>
                                <input type="text" class="form-control" id="page_id" name="page_id"
                                       value="{{ current_token.page_id if current_token else '' }}"
                                       placeholder="e.g., 123456789012345">
                            </div>

                            <div class="mb-3">
                                <label for="page_name" class="form-label">Page Name (Optional)</label>
                                <input type="text" class="form-control" id="page_name" name="page_name"
                                       value="{{ current_token.page_name if current_token else '' }}"
                                       placeholder="e.g., My Business Page">
                            </div>

                            <div class="d-grid gap-2">
                                <button type="submit" class="btn btn-primary py-2">Save Token</button>
                                <a href="{{ url_for('dashboard') }}" class="btn btn-outline-secondary py-2">Cancel</a>

                                {% if current_token %}
                                <a href="{{ url_for('facebook_delete_token', ai_id=ai_id) }}"
                                   class="btn btn-outline-danger py-2"
                                   onclick="return confirm('Are you sure you want to delete your Facebook token?')">
                                    Delete Token
                                </a>
                                {% endif %}
                            </div>
                        </form>

                        <hr>

                        <div class="mt-3">
                            <h5>How to get your Facebook Access Token:</h5>
                            <ol class="small">
                                <li>Go to <a href="https://developers.facebook.com/" target="_blank">Facebook Developers</a></li>
                                <li>Create or select your App</li>
                                <li>Go to Tools → Graph API Explorer</li>
                                <li>Select your Page from the dropdown</li>
                                <li>Copy the Access Token</li>
                                <li>Paste it in the field above</li>
                            </ol>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
'''

# Add this template with your other templates
ANALYSIS_RESULTS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Results - AI Workforce</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <h2>Facebook Analysis Results</h2>

        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Performance Metrics</h5>
                    </div>
                    <div class="card-body">
                        <p><strong>Total Posts:</strong> {{ analysis.total_posts }}</p>
                        <p><strong>Total Likes:</strong> {{ analysis.total_likes }}</p>
                        <p><strong>Total Comments:</strong> {{ analysis.total_comments }}</p>
                        <p><strong>Total Shares:</strong> {{ analysis.total_shares }}</p>
                        <p><strong>Avg Engagement:</strong> {{ "%.2f"|format(analysis.avg_engagement_rate) }}</p>
                    </div>
                </div>
            </div>

            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Top Performing Posts</h5>
                    </div>
                    <div class="card-body">
                        {% for post in analysis.top_performing_posts %}
                            <div class="mb-2">
                                <small>{{ post.message }}</small><br>
                                <span class="text-muted">Engagement: {{ post.engagement }}</span>
                            </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>

        <div class="card mt-4">
            <div class="card-header">
                <h5>AI Recommendations</h5>
            </div>
            <div class="card-body">
                <pre style="white-space: pre-wrap;">{{ ai_analysis }}</pre>
            </div>
        </div>

        <div class="mt-3">
            <a href="{{ url_for('dashboard') }}" class="btn btn-primary">Back to Dashboard</a>
            <a href="/facebook/generate_content/{{ ai_id }}?topic=based on analysis" class="btn btn-success">Generate Content Ideas</a>
        </div>
    </div>
</body>
</html>
'''
# Add this to the dashboard template where needed
# Add this function to handle the sidebar toggle
def add_sidebar_js():
    return SIDEBAR_JS

# Update the dashboard template to include the JavaScript
DASHBOARD_TEMPLATE = DASHBOARD_TEMPLATE.replace('</body>', f'{SIDEBAR_JS}</body>')

# Main application entry point
if __name__ == '__main__':
    # Initialize JSON files
    init_json_files()

    # Create upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Run the application
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    app.run(host='0.0.0.0', port=port, debug=True)