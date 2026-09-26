from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from models.db_models import db, User, DigitalTwin
from utils.helpers import login_required, get_current_user

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard.dashboard_view'))

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        age = request.form.get('age', type=int)
        gender = request.form.get('gender', 'Male')
        occupation = request.form.get('occupation', 'Other')

        # Validation
        if not full_name or not email or not password:
            flash('Please fill in all required fields.', 'danger')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return render_template('register.html')

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('An account with this email address already exists. Please log in.', 'warning')
            return redirect(url_for('auth.login'))

        # Create user
        new_user = User(
            full_name=full_name,
            email=email,
            age=age or 30,
            gender=gender,
            occupation=occupation
        )
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        # Initialize Digital Twin for user
        twin = DigitalTwin(
            user_id=new_user.id,
            baseline_health_score=0.0,
            total_assessments=0
        )
        db.session.add(twin)
        db.session.commit()

        session['user_id'] = new_user.id
        session['user_name'] = new_user.full_name
        flash(f'Welcome, {new_user.full_name}! Your account and Digital Twin have been initialized.', 'success')
        return redirect(url_for('dashboard.dashboard_view'))

    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard.dashboard_view'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Please provide both email and password.', 'danger')
            return render_template('login.html')

        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash('Invalid email or password. Please try again.', 'danger')
            return render_template('login.html')

        session.clear()
        session['user_id'] = user.id
        session['user_name'] = user.full_name

        flash(f'Welcome back, {user.full_name}!', 'success')
        next_page = request.args.get('next')
        if next_page and next_page.startswith('/'):
            return redirect(next_page)
        return redirect(url_for('dashboard.dashboard_view'))

    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    session.clear()
    flash('You have been safely logged out.', 'info')
    return redirect(url_for('auth.login'))


@auth_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = get_current_user()
    if request.method == 'POST':
        user.full_name = request.form.get('full_name', user.full_name).strip()
        user.age = request.form.get('age', user.age, type=int)
        user.gender = request.form.get('gender', user.gender)
        user.occupation = request.form.get('occupation', user.occupation)

        new_password = request.form.get('new_password', '').strip()
        if new_password:
            if len(new_password) < 6:
                flash('New password must be at least 6 characters.', 'warning')
            else:
                user.set_password(new_password)
                flash('Password updated successfully.', 'success')

        db.session.commit()
        session['user_name'] = user.full_name
        flash('Profile updated successfully.', 'success')
        return redirect(url_for('auth.profile'))

    return render_template('profile.html', user=user)
