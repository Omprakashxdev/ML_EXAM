

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:prilim_exam/login.dart';

class Splashscreen extends StatefulWidget {
  const Splashscreen({super.key});

  @override
  State<Splashscreen> createState() => _SplashscreenState();
}

class _SplashscreenState extends State<Splashscreen> {

//   @override
// void  initState(){
//   super.initState();
  
//   Timer(Duration(seconds: 5), (){
//     Navigator.pushReplacement(context, MaterialPageRoute(builder: (context)=>Login()));
//   });

// }
@override
void initState(){
  super.initState(

  );
Timer(Duration(seconds: 1), (){
  Navigator.pushReplacement(context, MaterialPageRoute(builder: (conrext)=>Login()));
});
}
  @override
  Widget build(BuildContext context) {
    

    return Scaffold(
      body: Center(
        child: Text("hello world ",style: TextStyle(
          fontWeight: FontWeight.bold,
          color: Colors.blueAccent,
          fontStyle: FontStyle.italic,
          fontSize: 
          50.0
        ),),
      ),
    
    );
  }
}