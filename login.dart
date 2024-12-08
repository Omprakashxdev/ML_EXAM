import 'package:flutter/material.dart';

void main(){
  runApp(MaterialApp(
    home: Login(),
  ));
}

class Login extends StatelessWidget {
  const Login({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.black,
      centerTitle: true,
      title: Text("whattsapp",
      style: TextStyle(
        fontWeight: FontWeight.bold,
          color: Colors.blueAccent,
          fontStyle: FontStyle.italic,
          fontSize: 
          50.0
      ),
      ),
      // leading: IconButton(onPressed: (){}, icon: Icon(Icons.menu), color: Colors.amber,),
      actions: [
        IconButton(onPressed: (){}, icon: Icon(Icons.more_vert), color: Colors.redAccent,)
      ],
      
      ),
      drawer: Drawer(
        backgroundColor: Color.fromARGB(255, 255, 255, 255),
        child: ListView(

          children: [
            UserAccountsDrawerHeader(accountName: Text("geoigjeogjre"),
            accountEmail: Text("fnn@123"),
            currentAccountPicture:CircleAvatar(
              backgroundImage: NetworkImage("https://i.pinimg.com/200x150/84/35/da/8435da278ca3ad44527dbb5d4c256175.jpg"),
              ),
            ),
            SizedBox(height: 20,),
            ListTile(
              onTap: (){
              
            },
            leading: Icon(Icons.home),
            title: Text("Home"),

          
            ),
            Divider(
              height: 1.0,

            ),//d
            
            SizedBox(height: 20,),
            ListTile(
              onTap: (){
              
              },
              leading: Icon(Icons.account_circle),
              title: Text("Profile"),
            ),
            SizedBox(height: 20,),
            ListTile(
              onTap: (){
              
            },
            leading: Icon(Icons.kayaking),
            title: Text("boat"),

          
            ),
            SizedBox(height: 20,),
            ListTile(
              onTap: (){
              
            },
            leading: Icon(Icons.offline_bolt),
            title: Text("bolts"),

          
            ),
            Divider(
              height: 1.0,

            ),//d
            Divider(
              height: 1.0,

            ),//d
          ],
        ),
        
      
//       drawer: Drawer(
//         backgroundColor: Color.fromARGB(255, 255, 255, 255),
//         child: ListView(
//           children: [
//             UserAccountsDrawerHeader(accountName: Text("om"), 
//             accountEmail:Text("om@123"),
//             currentAccountPicture: CircleAvatar(
//           backgroundImage: NetworkImage("https://i.pinimg.com/200x150/84/35/da/8435da278ca3ad44527dbb5d4c256175.jpg"),
//             ),
//             ),
//             SizedBox(height: 20,),
//             ListTile(
//               onTap: (){

//               },
//               leading: Icon(Icons.home),
//               title: Text("Home"),
            
//             ),
//             Divider(
//               height: 1.0,

//             ),//d
// SizedBox(height: 20,),
//             ListTile(
//               onTap: (){
              
//             },
//             leading: Icon(Icons.account_circle),
//             title: Text("Profile"),

//             ),
//           ],
//         ),
//       ),

    ),
    );

    
  }
}