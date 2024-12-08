// ignore_for_file: prefer_const_constructors, prefer_const_literals_to_create_immutables

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:prilim_exam/login.dart';
import 'package:prilim_exam/whatstapbar.dart';
//import 'mainpage.dart';

class Splashscen extends StatefulWidget {
  const Splashscen({super.key});

  @override
  State<Splashscen> createState() => _SplashscenState();
}


class _SplashscenState extends State<Splashscen> {

  @override
  void initState() {
    
    super.initState();

    Timer(
         Duration(seconds: 5), () {
      Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => Tab_bar(),
          ));
    });
  }
 @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF25D366), // WhatsApp background color
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Spacer(), 
          Center(
            child: Image.asset(
              'assets/images/whtas.png',
              width: 150.0,
              height: 150.0,
            ),
          ),
          Spacer(), 
          Padding(
            padding: const EdgeInsets.only(bottom: 10.0),
            child: Text(
              'from ',
              style: TextStyle(
                color: Colors.white,
                fontSize: 16.0,
                fontWeight: FontWeight.w500,
                
              ),
            ),
            
          ),
          Padding(
            padding: const EdgeInsets.only(bottom: 20),
            child: Text("meta ",style: TextStyle(fontSize: 20,fontWeight: FontWeight.bold,color: Colors.white),),
          )
        ],
      ),
    );
  }
}