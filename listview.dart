import 'package:flutter/material.dart';

class Listview extends StatefulWidget {
  const Listview({super.key});

  @override
  State<Listview> createState() => _ListviewState();
}

class _ListviewState extends State<Listview> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("listview"),
      ),
      body: ListView
      (
        children: [
           ListTile(
            title: Text("List Item 1"),
            subtitle: Text("JUST SENT A MESSAGE"),
            tileColor: Colors.deepOrange,
            leading: Icon(Icons.person_pin),
            trailing: Icon(Icons.arrow_forward_ios) ,
            hoverColor: Colors.blue,
            onTap: () => {
              print("List Item 1 is tapped"),
             

              
            }, 
        ),
    SizedBox(height: 30,),

ListTile(
            title: Text("List Item 1"),
            subtitle: Text("JUST SENT A MESSAGE"),
            tileColor: Colors.deepOrange,
            leading: Icon(Icons.person_pin),
            trailing: Icon(Icons.arrow_forward_ios) ,
            hoverColor: Colors.blue,
            onTap: () => {
              print("List Item 1 is tapped"),
             

              
            }, 
        ),
    SizedBox(height: 30,),

ListTile(
            title: Text("List Item 1"),
            subtitle: Text("JUST SENT A MESSAGE"),
            tileColor: Colors.deepOrange,
            leading: Icon(Icons.person_pin),
            trailing: Icon(Icons.arrow_forward_ios) ,
            hoverColor: Colors.blue,
            onTap: () => {
              print("List Item 1 is tapped"),
             

              
            }, 
        ),
    SizedBox(height: 30,),

ListTile(
            title: Text("List Item 1"),
            subtitle: Text("JUST SENT A MESSAGE"),
            tileColor: Colors.deepOrange,
            leading: Icon(Icons.person_pin),
            trailing: Icon(Icons.arrow_forward_ios) ,
            hoverColor: Colors.blue,
            onTap: () => {
              print("List Item 1 is tapped"),
             

              
            }, 
        ),
    SizedBox(height: 30,),


ListTile(
            title: Text("List Item 1"),
            subtitle: Text("JUST SENT A MESSAGE"),
            tileColor: Colors.deepOrange,
            leading: Icon(Icons.person_pin),
            trailing: Icon(Icons.arrow_forward_ios) ,
            hoverColor: Colors.blue,
            onTap: () => {
              print("List Item 1 is tapped"),
             

              
            }, 
        ),
    SizedBox(height: 30,),


        ]
      )
    );
  }
}