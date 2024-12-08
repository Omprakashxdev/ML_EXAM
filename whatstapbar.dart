// ignore: file_names
import 'package:flutter/material.dart';

class Tab_bar extends StatelessWidget {
  const Tab_bar({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(length: 3, 
    child: Scaffold(
      appBar: AppBar(
        bottom: const TabBar(
          tabs: [
            Tab(
              icon: Icon(Icons.home),
              child: Text(
                "home"
              ),
              ),

              Tab(
                icon: Icon(Icons.settings)),

              Tab(
                icon: Icon(Icons.person)),
          ],
        ),
        title: const Text("fullteer demo"),


),
body:
TabBarView(children: [
  Container(
    color: Colors.orangeAccent,
    child: const Icon(Icons.home) ,
  ),
  Container(
    color: Colors.redAccent,
    child:  const Icon(Icons.settings),
  ),
  Container(
    color: Colors.greenAccent,
    child:  const Icon(Icons.person),
    
  ),
    ],
    ),
    ),
      );
      
  }
}