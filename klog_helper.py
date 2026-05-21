# This file is a helper module for preprocessing KLog messages before they 
# are parsed by the Drain log parser. The `preprocess_klog_message` function 
# performs key-value pair analysis and replaces short values (< 25 characters) 
# with wildcards <*> to improve template generalization.

import re


def preprocess_klog_message(message):
    """
    Preprocess a klog message by replacing short values after '=' with wildcards.
    
    For every key=value or key="value" pair where the value is less than 25 characters,
    replace the value with <*> to help the parser generalize better.
    
    Examples:
        'version="v1.2.3"' -> 'version="<*>"'
        'mode=""' -> 'mode="<*>"'
        'ipFamily="IPv4"' -> 'ipFamily="<*>"'
        'err="very long error message that exceeds 25 characters..."' -> kept as is
    
    Args:
        message: The log message content string
        
    Returns:
        The preprocessed message with short values replaced by wildcards
    """
    if not message or not isinstance(message, str):
        return message
    
    # Pattern 1: key="value" (quoted values)
    # Match key="value" where value doesn't contain quotes
    def replace_quoted_kv(match):
        key = match.group(1)
        value = match.group(2)
        
        # Only replace if value is short (< 25 characters)
        if len(value) < 50 and len(value) > 0:
            return f'{key}="<*>"'
        return match.group(0)
    
    # Pattern 2: key=value (unquoted values)
    # Match key=value where value is alphanumeric, dots, slashes, hyphens, underscores
    def replace_unquoted_kv(match):
        key = match.group(1)
        value = match.group(2)
        
        # Only replace if value is short (< 50 characters)
        if len(value) < 50 and len(value) > 0:
            return f'{key}=<*>'
        return match.group(0)
    
    # First handle quoted values: key="value"
    result = re.sub(r'(\w+)="([^"]*)"', replace_quoted_kv, message)
    
    # Then handle unquoted values: key=value (but not if already processed)
    # Only match if not followed by another quote (to avoid double-processing)
    result = re.sub(r'(\w+)=([a-zA-Z0-9_./-]+)(?!\w)', replace_unquoted_kv, result)
    
    return result


def preprocess_klog_content(content):
    """
    Alias for preprocess_klog_message for compatibility with Drain parser.
    """
    return preprocess_klog_message(content)