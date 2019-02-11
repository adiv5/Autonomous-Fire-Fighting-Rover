using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class randompos : MonoBehaviour
{	public Vector3[] positions;
    // Start is called before the first frame update
    void Start()
    {
       float x = Random.Range(10, 150);
       float z = Random.Range(10, 150);
       Debug.LogFormat("coords = {0},{1}",x,z);

       Vector3 position = new Vector3(x, 0f, z);
      //
      //Instantiate(prefab, position, Quaternion.identity);
      transform.position=position;   
    }

}
