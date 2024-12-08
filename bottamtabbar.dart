import 'package:flutter/material.dart';

class Bottamtabbar extends StatefulWidget {
  const Bottamtabbar({super.key});

  @override
  State<Bottamtabbar> createState() => _BottamtabbarState();
}

class _BottamtabbarState extends State<Bottamtabbar> {

int _index=0;
List<Widget>  tapu =[
  Icon(Icons.home),
  Icon(Icons.settings),
  Icon(Icons.person),
  Icon(Icons.real_estate_agent_outlined),
  Icon(Icons.portable_wifi_off_outlined),

];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // body: Center(
      //   child: tapu[_index],
        
      // ),
      bottomNavigationBar: BottomNavigationBar(
           onTap: (index){
        setState(() {
                  _index=index; 

        });
        
      },
       currentIndex: _index ,
      items: [
        BottomNavigationBarItem(icon: Icon(Icons.home, color: Colors.red,), label: "home"),
               BottomNavigationBarItem(icon: Icon(Icons.home, color: Colors.red,), label: "home"),

        BottomNavigationBarItem(icon: Icon(Icons.home, color: Colors.red,), label: "home"),


        BottomNavigationBarItem(icon: Icon(Icons.home, color: Colors.red,), label: "home"),

        BottomNavigationBarItem(icon: Icon(Icons.home, color: Colors.red,), label: "home"),


      ],
   
      ),
    );
  }
}