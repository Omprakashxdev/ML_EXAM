import 'package:flutter/material.dart';

class DrawerandactionScreen extends StatelessWidget {
  const DrawerandactionScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false, // Hides the debug banner
      title: 'Home Page with Drawer and Action Menu',
      theme: ThemeData(
        primarySwatch: Colors.blue, // Sets the primary theme color
      ),
      home: HomePage(), // Sets HomePage as the default screen
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Home Page"), // Title of the AppBar
        actions: [
          // Action menu items (icons) in the AppBar
          IconButton(
            icon: Icon(Icons.search), // Search icon
            onPressed: () {
              // Handle search action here
              print("Search button clicked");
            },
          ),
          IconButton(
            icon: Icon(Icons.notifications), // Notifications icon
            onPressed: () {
              // Handle notification action here
              print("Notifications button clicked");
            },
          ),
          IconButton(
            icon: Icon(Icons.more_vert), // More options icon
            onPressed: () {
              // Handle more action here
              print("More options clicked");
            },
          ),
        ],
      ),
      drawer: Drawer(
        child: ListView(
          children: <Widget>[
            UserAccountsDrawerHeader(
              accountName: Text("Aesha"),
              accountEmail: Text("aesha@gmail.com"),
              currentAccountPicture: CircleAvatar(
                  backgroundImage: NetworkImage(
                      "https://thumbs.dreamstime.com/b/d-avatar-illustration-smiling-happy-girl-cartoon-close-up-portrait-standing-isolated-transparent-png-background-generative-272798686.jpg")),
            ),
            // Drawer Header
            DrawerHeader(
              decoration: BoxDecoration(
                color: Colors.blue, // Background color of the drawer header
              ),
              child: Text(
                'Drawer Menu',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                ),
              ),
            ),
            // Drawer menu items
            ListTile(
              leading: Icon(Icons.home), // Icon for the home item
              title: Text('Home'), // Title for the home item
              onTap: () {
                // Handle Home tap action here
                Navigator.pop(context); // Closes the drawer
              },
            ),
            ListTile(
              leading: Icon(Icons.settings), // Icon for the settings item
              title: Text('Settings'), // Title for the settings item
              onTap: () {
                // Handle Settings tap action here
                Navigator.pop(context); // Closes the drawer
              },
            ),
            ListTile(
              leading: Icon(Icons.contact_mail), // Icon for the contact item
              title: Text('Contact Us'), // Title for the contact item
              onTap: () {
                // Handle Contact tap action here
                Navigator.pop(context); // Closes the drawer
              },
            ),
          ],
        ),
      ),
      body: Center(
        // The main content of the home page
        child: Text(
          "Welcome to the Home Page!",
          style: TextStyle(fontSize: 24),
        ),
      ),
    );
  }
}
