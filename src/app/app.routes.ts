import { Routes } from '@angular/router';
import { LoginComponent } from './login/login.component';
import { ProfileComponent } from './profile/profile.component';
import { HomeComponent } from './home/home.component';
import { ReactiveformComponent } from './reactiveform/reactiveform.component';

export const routes: Routes = [

    {
        component:LoginComponent,
        path:'login'

    },
    {
        component:ProfileComponent,
        path:'Home'
    },
    { path: '', redirectTo: '/login', pathMatch: 'full' },
     {
        component:HomeComponent,
        path:'home'
     },
   
     {
        component:ReactiveformComponent,
        path:'Form'
     }


];



