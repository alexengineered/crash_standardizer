"""
AI Configuration Module
Handles API settings persistence (config file) and AI provider calls.
"""

import json
from pathlib import Path
from dataclasses import dataclass


# Config file location (same directory as app)
CONFIG_FILE = Path(__file__).parent / ".ai_config.json"

# Provider configurations
PROVIDERS = {
    "anthropic": {
        "name": "Anthropic (Claude)",
        "model": "claude-sonnet-4-20250514",
        "description": "Best for technical docs, most accurate"
    },
    "openai": {
        "name": "OpenAI (GPT-4)",
        "model": "gpt-4o-mini",
        "description": "Strong reasoning and versatile"
    },
    "google": {
        "name": "Google (Gemini)",
        "model": "gemini-1.5-flash",
        "description": "Balance of quality and speed"
    }
}


@dataclass
class AIConfig:
    """Stored AI configuration."""
    provider: str = ""
    api_key: str = ""
    
    @property
    def is_configured(self) -> bool:
        return bool(self.provider and self.api_key)
    
    @property
    def provider_name(self) -> str:
        if self.provider in PROVIDERS:
            return PROVIDERS[self.provider]["name"]
        return ""


def load_config() -> AIConfig:
    """Load AI config from file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                return AIConfig(
                    provider=data.get("provider", ""),
                    api_key=data.get("api_key", "")
                )
        except (json.JSONDecodeError, IOError):
            pass
    return AIConfig()


def save_config(config: AIConfig) -> bool:
    """Save AI config to file. Returns True on success."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({
                "provider": config.provider,
                "api_key": config.api_key
            }, f)
        return True
    except IOError:
        return False


def clear_config() -> bool:
    """Remove saved config. Returns True on success."""
    try:
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
        return True
    except IOError:
        return False


def call_ai(provider: str, api_key: str, prompt: str) -> str:
    """
    Call AI provider with prompt.
    
    Args:
        provider: Provider key ("anthropic", "openai", "google")
        api_key: API key for the provider
        prompt: The prompt to send
        
    Returns:
        Response text from the AI
        
    Raises:
        Exception on API errors
    """
    if provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=PROVIDERS["anthropic"]["model"],
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=PROVIDERS["openai"]["model"],
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    elif provider == "google":
        import requests
        model = PROVIDERS["google"]["model"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        response = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}]
        })
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    
    raise ValueError(f"Unknown AI provider: {provider}")


def test_api_key(provider: str, api_key: str) -> tuple[bool, str]:
    """
    Test if API key is valid with a simple request.
    
    Returns:
        (success, message) tuple
    """
    try:
        response = call_ai(provider, api_key, "Reply with only the word 'OK'")
        return True, "API key verified successfully"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "invalid" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return False, "Invalid API key"
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return False, "API quota exceeded"
        else:
            return False, f"API error: {error_msg[:100]}"
