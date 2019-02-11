using UnityEngine;
using System.Collections;
using System.IO;
using System;

public class SS1 : MonoBehaviour {
    // Capture frames as a screenshot sequence. Images are
    // stored as PNG files in a folder - these can be combined into
    // a movie using image utility software (eg, QuickTime Pro).
    // The folder to contain our screenshots.
    // If the folder exists we will append numbers to create an empty folder.
    string folder = "myscreenshots";
    int frameRate = 15;
        
    void Start () {
        // Set the playback framerate (real time will not relate to game time after this).
        Time.captureFramerate = frameRate;
        Debug.LogFormat((Time.captureFramerate).ToString());
         System.IO.Directory.CreateDirectory(folder);
         

        StartCoroutine(CaptureScreen());
 }
        // Create the folder
       
            // Determine whether the directory exists.
            
public IEnumerator CaptureScreen()
 {
    // Wait till the last possible moment before screen rendering to hide the UI
    yield return null;
    GameObject.Find("Canvas").GetComponent<Canvas>().enabled = false;
 
    // Wait for screen rendering to complete
    yield return new WaitForEndOfFrame();
 
    // Take screenshot

        string name = string.Format("{0}/screenshots/screen_{1}x{2}_{3}.png", 
                              Application.dataPath, 
                              640, 480, 
                              System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss") );
        
        // Capture the screenshot to the specified file.
        ScreenCapture.CaptureScreenshot(name);
 
    // Show UI after we're done
    GameObject.Find("Canvas").GetComponent<Canvas>().enabled = true;
 }
}
    
    