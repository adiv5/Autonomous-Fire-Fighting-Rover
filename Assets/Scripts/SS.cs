using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
public class SS : MonoBehaviour
{    public string fileName = "robotcoords.txt";
	

     public int resWidth = 640; 
     public int resHeight = 480;
 	 public int fps=15;
 	 private float videoTime=0;
     private bool takeHiResShot = false;
     private bool robotlocupdate = false;
 	 protected static List<List<float>> coords = new List<List<float>>();
     public static string ScreenShotName(int width, int height) {
         return string.Format("{0}/screenshots/screen_{1}x{2}_{3}.png", 
                              Application.dataPath, 
                              width, height, 
                              System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss.ffff"));
     }
 
     public void TakeHiResShot() {
         takeHiResShot = true;
     }

void Start()
{
        if (File.Exists(fileName))
        {
            Debug.Log(fileName+" already exists.");
            return;
        } 
            
           Vector3 pos=GameObject.Find("FirstPersonCharacter").transform.position;
            File.AppendAllText(@"D:\FY project\Environment\Rover\robotcoords.txt", "X="+pos.x +  " Z="+pos.z + Environment.NewLine);
		   //sr.WriteLine ("X= {0} Z= {1}.",pos.x, pos.z);
           //sr.Close();
           List<float> temp=new List<float>();
           temp.Add(pos.x);
           temp.Add(pos.z);
           coords.Add(temp);
           Debug.Log(coords.Count);
        }
     void Update(){
            videoTime += Time.deltaTime;
            //Debug.LogFormat(((float)fps).ToString());
 
            if (videoTime >= (1f / (float)fps)) {
            	//Debug.LogFormat(videoTime.ToString());
                videoTime = 0;
                takeHiResShot=false;
                robotlocupdate=true;

                if(robotlocupdate){
                	Vector3 pos=GameObject.Find("FirstPersonCharacter").transform.position;


 				
             
                List<float> temp=new List<float>();
                temp.Add(pos.x);
                temp.Add(pos.z);
                coords.Add(temp);
           		 //Debug.Log("Updated count is "+coords.Count.ToString());
                File.AppendAllText(@"D:\FY project\Environment\Rover\robotcoords.txt", "X="+pos.x +  " Z="+pos.z + Environment.NewLine);
              
 
                	//Debug.LogFormat("{0},{1}",pos.x,pos.z);
                }

                 if (takeHiResShot) {
             RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);
             GetComponent<Camera>().targetTexture = rt;
             Texture2D screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
             GetComponent<Camera>().Render();
             RenderTexture.active = rt;
             screenShot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);
             GetComponent<Camera>().targetTexture = null;
             RenderTexture.active = null; // JC: added to avoid errors
             Destroy(rt);
             byte[] bytes = screenShot.EncodeToPNG();
             string filename = ScreenShotName(resWidth, resHeight);
             System.IO.File.WriteAllBytes(filename, bytes);
             Debug.Log(string.Format("Took screenshot to: {0}", filename));
             takeHiResShot = false;
         }
     }
 }

    /* void LateUpdate() {
         //takeHiResShot |= Input.GetKeyDown("k");
        
     }*/
 }

 