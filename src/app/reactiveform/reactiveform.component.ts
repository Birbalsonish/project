import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { ReactiveFormsModule } from '@angular/forms';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { HomeComponent } from '../home/home.component';

@Component({
  selector: 'reactiveform',
  standalone: true,
  imports: [ReactiveFormsModule,CommonModule,HomeComponent],
  templateUrl: './reactiveform.component.html',
  styleUrl: './reactiveform.component.css'
})
export class ReactiveformComponent {
  reactiveform: FormGroup;

  constructor() {
    this.reactiveform = new FormGroup({
      firstname: new FormControl('', [Validators.required]),
      lastname: new FormControl('', [Validators.required, Validators.minLength(4)]),
      email: new FormControl('', [Validators.required, Validators.email]),
      city: new FormControl(''),
      state: new FormControl(''),
      zipcode: new FormControl(''),
      isagree: new FormControl(false, [Validators.requiredTrue])
    });
  }

  onSubmit(){
  const isformvalid=this.reactiveform.valid;
  debugger;
}

generateusername(){
  let username:string='';
   const fname:string=this.reactiveform.get('firstname')?.value;
   const lname:string=this.reactiveform.get('lastname')?.value;

   if(fname.length >= 3){
    username += fname.slice(0,3);
   }
   else{
    username+=fname;
   }
   if(lname.length >= 3){
    username  += lname.slice(0,3);
   }
   else{
    username+=lname;
   }
   console.log(username);


}

} 

