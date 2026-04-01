document.addEventListener('DOMContentLoaded', function() {
    // Stats animation
    const observerOptions = { threshold: 0.1, rootMargin: '0px 0px -50px 0px' };
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    });
    
    document.querySelectorAll('.stat-box').forEach((box, index) => {
        box.style.opacity = '0';
        box.style.transform = 'translateY(30px)';
        box.style.transitionDelay = `${index * 0.1}s`;
        observer.observe(box);
    });
    
    // Form feedback
    document.querySelector('form').addEventListener('submit', function(e) {
        const btn = document.querySelector('.btn-primary');
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
        btn.disabled = true;
    });
    
    // Copy result
    document.querySelectorAll('.alert').forEach(alert => {
        alert.addEventListener('click', function() {
            navigator.clipboard.writeText(this.textContent);
            const icon = this.querySelector('i');
            const oldIcon = icon.className;
            icon.className = 'fas fa-copy fa-2x float-start me-3 text-success';
            setTimeout(() => icon.className = oldIcon, 1000);
        });
    });
});