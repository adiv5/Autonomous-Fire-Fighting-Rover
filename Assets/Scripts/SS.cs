using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SS : MonoBehaviour
{
     public int resWidth = 640; 
     public int resHeight = 480;
 	 public int fps=15;
 	 private float videoTime=0;
     private bool takeHiResShot = false;
     private bool robotlocupdate = false;
 
     public static string ScreenShotName(int width, int height) {
         return string.Format("{0}/screenshots/screen_{1}x{2}_{3}.png", 
                              Application.dataPath, 
                              width, height, 
                              System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss.ffff"));
     }
 
     public void TakeHiResShot() {
         takeHiResShot = true;
     }
     void Update(){
            videoTime += Time.deltaTime;
            Debug.LogFormat(((float)fps).ToString());
 
            if (videoTime >= (1f / (float)fps)) {
            	//Debug.LogFormat(videoTime.ToString());
                videoTime = 0;
                takeHiResShot=true;
                robotlocupdate=true;

                if(robotlocupdate){
                	Vector3 pos=GameObject.Find("FirstPersonCharacter").transform.position;
                	Debug.Log(pos);
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

 