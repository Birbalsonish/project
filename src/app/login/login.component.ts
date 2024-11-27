import { Component, inject } from '@angular/core';
import { FormsModule, NgForm } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { AuthService } from '../service/auth.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [FormsModule, HttpClientModule],
  templateUrl: './login.component.html',
  styleUrl: './login.component.css',
  providers:[AuthService]

})
export class LoginComponent {
  authservice: AuthService = inject(AuthService);
  isSignUpMode = false; // Initial mode is Log In
  router: Router = inject (Router)

  toggleMode(event: Event): void {
    event.preventDefault(); // Prevent default link behavior
    this.isSignUpMode = !this.isSignUpMode; // Toggle the mode
  }

  onformsumbited(form: NgForm): void {
    const { email, password } = form.value; // Destructure form values

    if (this.isSignUpMode) {
      this.authservice.signup(email, password).subscribe({
        next: (res) => console.log('Sign Up Successful:', res),
        error: (err) => console.error('Sign Up Error:', err),
      });
    } else { 
       this.authservice.login(email, password).subscribe({
        next: (res) => {console.log('Log In Successful:', res)
          this.router.navigate(['/home']);
          alert('you are logged in');
        },
        error: (err) => console.error('Sign Up Error:', err),
      });

  }
  
}
}
