import { Injectable, inject } from '@angular/core';
import {UserService} from '../service/profileservice'

@Injectable({
  providedIn: 'root'
})
export class Authgaurds {
  isLogged: boolean = false;
  userservice:UserService = inject(UserService)

  login(username: string, password: string) {
    const user = this.userservice.users.find(( u:any )=> u.username === username && u.password === password);

    if (user === undefined) {
      this.isLogged = false;
    } else {
      this.isLogged = true;
    }

    return user;
  }

  logout() {
    this.isLogged = false;
  }
}