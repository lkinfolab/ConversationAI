from django.shortcuts import render


# ── Auth ──────────────────────────────────────────────────────────────────────

def login_view(request):
    return render(request, 'auth/login.html')


def register_view(request):
    return render(request, 'auth/register.html')


# ── User ──────────────────────────────────────────────────────────────────────

def user_dashboard(request):
    return render(request, 'user/dashboard.html')


def user_projects(request):
    return render(request, 'user/projects.html')


def user_project_detail(request, project_id):
    return render(request, 'user/project_detail.html', {'project_id': project_id})


def user_chat(request, conversation_id=None):
    return render(request, 'user/chat.html', {'conversation_id': conversation_id})


def user_responses(request):
    return render(request, 'user/responses.html')


def user_profile(request):
    return render(request, 'user/profile.html')


# ── Moderator ─────────────────────────────────────────────────────────────────

def moderator_dashboard(request):
    return render(request, 'moderator/dashboard.html')


def moderator_queue(request):
    return render(request, 'moderator/queue.html')


def moderator_review(request, queue_id):
    return render(request, 'moderator/review.html', {'queue_id': queue_id})


def moderator_history(request):
    return render(request, 'moderator/history.html')


# ── Admin Panel ───────────────────────────────────────────────────────────────

def admin_dashboard(request):
    return render(request, 'admin_panel/dashboard.html')


def admin_users(request):
    return render(request, 'admin_panel/users.html')


def admin_projects(request):
    return render(request, 'admin_panel/projects.html')
