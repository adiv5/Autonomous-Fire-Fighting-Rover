using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;
using Newtonsoft.Json;

public class floorplan : MonoBehaviour
{	public Vector3[] positions;
  public string fileName = "floorplan.txt";
  GameObject controller;
  
  public  List<Tuple<float,float,string>> pos = new List<Tuple<float,float,string>>();
  
  //protected static List<List<float,float,string>> coords = new List<List<float,float,string>>();
    // Start is called before the first frame update
    void Start()
    { 
       float x = UnityEngine.Random.Range(10, 200);
       float z = UnityEngine.Random.Range(10, 200);
       controller=GameObject.Find("FirstPersonCharacter");
       //Debug.LogFormat(this.transform.name); 
       pos.Add(new Tuple<float, float,string>(x,z,this.transform.tag));
       File.AppendAllText(@"D:\FY project\Environment\Rover\floorplan.txt", "coords="+x+","+z+  "   class="+this.transform.tag + Environment.NewLine);
       //Debug.LogFormat("coords,object = {0},{1},{2}",x,z,this.transform.tag);


       Vector3 position = new Vector3(x, 0f, z);
      //
      
      transform.position=position;   
    }

}
