from django.core.cache import cache
from django.contrib import messages
from django.shortcuts import redirect
from django.contrib.auth import authenticate
from django.http import HttpResponse, JsonResponse
import time
import re

class SecurityMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Rate limiting for login attempts
        if request.path == '/login/' and request.method == 'POST':
            ip = request.META.get('REMOTE_ADDR')
            attempts_key = f'login_attempts_{ip}'
            timeout_key = f'login_timeout_{ip}'
            
            # Check if user is in timeout
            if cache.get(timeout_key):
                messages.error(request, 'Too many failed login attempts. Please try again in 5 minutes.')
                return redirect('login')
            
            # Track attempts
            attempts = cache.get(attempts_key, 0)
            if attempts >= 5:  # Max 5 attempts
                cache.set(timeout_key, True, 300)  # 5 minute timeout
                cache.delete(attempts_key)
                messages.error(request, 'Account temporarily locked due to multiple failed attempts.')
                return redirect('login')

        # Rate limiting for registration
        if request.path == '/register/' and request.method == 'POST':
            ip = request.META.get('REMOTE_ADDR')
            reg_attempts_key = f'register_attempts_{ip}'
            attempts = cache.get(reg_attempts_key, 0)
            
            if attempts >= 3:  # Max 3 registration attempts per hour
                messages.error(request, 'Too many registration attempts. Please try again later.')
                return redirect('register')
            
            cache.set(reg_attempts_key, attempts + 1, 3600)  # 1 hour timeout

        response = self.get_response(request)
        
        # Clear login attempts on successful authentication
        if hasattr(request, 'user') and request.user.is_authenticated:
            ip = request.META.get('REMOTE_ADDR')
            cache.delete(f'login_attempts_{ip}')
            cache.delete(f'login_timeout_{ip}')

        return response

class LoginAttemptMiddleware:
    """Legacy middleware - keeping for backwards compatibility"""
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path == '/login/' and request.method == 'POST':
            ip = request.META.get('REMOTE_ADDR')
            attempts_key = f'login_attempts_{ip}'
            timeout_key = f'login_timeout_{ip}'
            
            # Check if user is in timeout
            if cache.get(timeout_key):
                messages.error(request, 'Too many failed attempts. Please try again later.')
                return redirect('login')
            
            # Track attempts
            attempts = cache.get(attempts_key, 0)
            if attempts >= 5:  # Max 5 attempts
                cache.set(timeout_key, True, 300)  # 5 minute timeout
                cache.delete(attempts_key)
                messages.error(request, 'Too many failed attempts. Please try again in 5 minutes.')
                return redirect('login')
            
            # Increment attempts counter
            cache.set(attempts_key, attempts + 1, 300)  # Reset after 5 minutes
            
        response = self.get_response(request)
        return response


class APIAuthenticationMiddleware:
    """Middleware to handle API authentication gracefully"""
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        
        # If this is an API request and we get a login redirect, return JSON instead
        if (request.path.startswith('/api/') and 
            response.status_code == 302 and 
            '/login/' in response.get('Location', '')):
            
            return JsonResponse({
                'success': False,
                'error': 'Authentication required',
                'message': 'Please log in to access this feature',
                'redirect_url': '/login/'
            }, status=401)
            
        return response
