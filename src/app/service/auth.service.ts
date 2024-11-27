import { Injectable,inject } from "@angular/core";
import { HttpClient } from "@angular/common/http";
import { Authresponse } from "./Authresponse";
import { catchError, throwError } from "rxjs";


@Injectable({
    providedIn:'root'
})
export class AuthService{
    http:HttpClient=inject(HttpClient);


    signup(email:string ,password:string){
        const data={email: email, password:password,returnSecureToken:true};
        const apiKey = 'AIzaSyApDaZL01fO2X72oVJrO2bjy3RQhFryWS8';
       
       return this.http.post<Authresponse>(`https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=${apiKey}`,data)
       .pipe(
        catchError((err) => {
          let errorMessage = 'An unknown error has occurred';

          console.log('Error Response:', err);  // Debugging: Log the full error object

          // Ensure `err.error` exists before accessing it
          if (err.error && err.error.error) {
            switch (err.error.error.message) {
              case 'EMAIL_EXISTS':
                errorMessage = 'This email already exists.';
                break;
              case 'OPERATION_NOT_ALLOWED':
                errorMessage = 'This operation is not allowed.';
                break;
              default:
                errorMessage = err.error.error.message || errorMessage;
                break;
            }
          }

          // Return the error message to the subscriber
          return throwError(() => new Error(errorMessage));
        })
      );
  }
  login(email:string ,password:string){
    const data={email: email, password:password,returnSecureToken:true};
    const apiKey = 'AIzaSyApDaZL01fO2X72oVJrO2bjy3RQhFryWS8';
   
   return this.http.post<Authresponse>(`https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=${apiKey}`,data)
   .pipe(
    catchError((err) => {
      let errorMessage = 'An unknown error has occurred';

      console.log('Error Response:', err);  // Debugging: Log the full error object

      // Ensure `err.error` exists before accessing it
      if (err.error && err.error.error) {
        switch (err.error.error.message) {
          case 'EMAIL_NOT_FOUND':
            errorMessage = 'This email is not found.';
            alert('This email is not found');
            break;
          case 'INVALID_PASSWORD':
            errorMessage = 'This password is invalid.';
            alert('password is invalid')
            break;
          default:
            errorMessage = err.error.error.message || errorMessage;
            break;
        }
      }

      // Return the error message to the subscriber
      return throwError(() => new Error(errorMessage));
    })
  );
  }
}