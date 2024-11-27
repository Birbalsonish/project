import { Component } from '@angular/core';
import { ReactiveformComponent } from '../reactiveform/reactiveform.component';
import { HomeComponent } from '../home/home.component';



@Component({
  selector: 'app-profile',
  standalone: true,
  imports: [ReactiveformComponent,HomeComponent],
  templateUrl: './profile.component.html',
  styleUrl: './profile.component.css'
})
export class ProfileComponent {

}
