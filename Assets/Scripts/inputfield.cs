using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.UI;
using System.IO;
using System;

public class inputfield : MonoBehaviour {
	public string fileName = "sentences.txt";
    void Start ()
    {
        var input = gameObject.GetComponent<InputField>();
        //var se= new InputField.SubmitEvent();
        //se.AddListener(SubmitName);
        //input.onEndEdit = se;

        //or simply use the line below, 
        input.onEndEdit.AddListener(SubmitName);  // This also works
    }

    private void SubmitName(string arg0)
    {
    	File.AppendAllText(@"D:\FY project\Environment\Rover\sentences.txt", ""+arg0 + Environment.NewLine);
        Debug.Log(arg0);
    }

 


}

