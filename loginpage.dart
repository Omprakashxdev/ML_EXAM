import 'package:flutter/material.dart';
import 'package:prilim_exam/ragister.dart';

class Loginpage extends StatelessWidget {
  const Loginpage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.blue,
        title: Text("Logimpage"),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.only(top: 20.0,right: 30, left: 30),
        child: Center(
          child: Column(
            children: [
              Icon(Icons.home,size: 30, color: Colors.amber,),
              Text("Welcome to login page"),
              SizedBox(height: 20.0,),
              TextField(
                decoration: InputDecoration(
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10.0),

                  ),
                  hintText: "enter your name",
                  suffixIcon: Icon(Icons.cancel),
                  prefix: Icon(Icons.person),

                  labelText: "hiii "
                ),

              ),
              SizedBox(height: 20.0,),
              TextField( 
                decoration: InputDecoration(
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10.0),
                ),
                hintText: "enter your email",
                suffixIcon: Icon(Icons.cancel),
                prefix: Icon(Icons.email),

               ),

              ),
              Align(
                alignment: Alignment.bottomRight,
                child:  TextButton(onPressed: (){}, child: Text("forgot password",style: TextStyle(
                  color: Colors.blue,),
                  

                ),
                )
                ,),
                 SizedBox(height: 20.0),
            ElevatedButton(
              onPressed: () {},
              child: Text(
                "login",
                style: TextStyle(color: Colors.white),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: Color.fromARGB(255, 41, 199, 120),
                padding: EdgeInsets.symmetric(horizontal: 120),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(40),
                ),
              ),
            ),
            SizedBox(height: 20.0,),
ElevatedButton(onPressed: (){
  Navigator.push(context, MaterialPageRoute(builder: (context)=>ragister()));
},
 child: Text("sign up",style:TextStyle(color: Colors.black) ,),
 
 style: ElevatedButton.styleFrom(
  backgroundColor: Color.fromARGB(255, 16, 23, 27),
  foregroundColor:Color.fromARGB(255, 69, 111, 136) ,
  //padding: EdgeInsets.symmetric(horizontal: 120),
  shape: RoundedRectangleBorder(
    borderRadius: BorderRadius.circular(40),
  ),
 ),),

           SizedBox(height: 20.0,),
           Row(mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircleAvatar(
                    backgroundColor: Colors.amber,
                    radius: 30,
                    child: Icon(Icons.facebook),

                  )
             
           ])
                ],
              )
            // TextButton(onPressed: (){}, child: Text("forgot password")),
            
            
          ),
        ),
      );
  
  }
}