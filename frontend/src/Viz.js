import React, { Component } from "react";
// import ReactDOM from "react-dom";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
// import { Vector3 } from "three";
import './Viz.css'

class Viz extends Component {
  componentDidMount() {
    var scene = new THREE.Scene();
    // const canvas = document.querySelector('viewport');
    var camera = new THREE.PerspectiveCamera( 30, window.innerWidth/window.innerHeight, 0.1, 1000 );
    // var renderer = new THREE.WebGLRenderer({ canvas });
    var renderer = new THREE.WebGLRenderer( { antialias: true } );
    scene.background = new THREE.Color('grey');
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );
    renderer.render( scene, camera );

    camera.position.z = 4;
    camera.position.y = 2.5;
    camera.position.x = 3;
    camera.lookAt(new THREE.Vector3(0, 0, 0));
    const controls = new OrbitControls( camera, renderer.domElement );

    controls.minDistance = 0.5;
    controls.maxDistance = 8;

    var overhead = new THREE.AmbientLight( {color: 0xffffff}, 1);
    scene.add( overhead );

    const lightL1 = new THREE.DirectionalLight( 0x404040, 1); // soft white light
    const lightR1 = new THREE.DirectionalLight( 0x404040, 1); // soft white light
    const lightL2 = new THREE.DirectionalLight( 0x404040, 1); // soft white light
    const lightR2 = new THREE.DirectionalLight( 0x404040, 1); // soft white light
    lightL1.position.x = -1;
    lightR1.position.x = 1;
    lightL2.position.x = -1;
    lightR2.position.x = 1;
    lightL1.position.z = 1;
    lightR1.position.z = 1;
    lightL2.position.z = -1;
    lightR2.position.z = -1;
    scene.add( lightL1 );
    scene.add( lightR1 );
    scene.add( lightL2 );
    scene.add( lightR2 );

    var x = -1
    var ballMesh = new THREE.SphereGeometry( 0.02 );
    // var ballMat = new THREE.MeshBasicMaterial( { color: 0xdadada } );
    var ballMat = new THREE.MeshLambertMaterial( { color: 0xdadada, emissive: 0xdadada, emissiveIntensity: .5} );
    for (var i = 0; i < 20; i++) {
      window['cube'+i] = new THREE.Mesh( ballMesh, ballMat );
      eval('cube'+i).position.x = ( x+(i/15) );
      eval('cube'+i).position.y = ( i/15 );
      // eval('cube'+i).scale.set(i/10, i/10, i/10);
      eval('cube'+i).castShadow = true;
      eval('cube'+i).receiveShadow = true;
      scene.add( eval('cube'+i) );
      // console.log(eval('cube'+i));
      renderer.render( scene, camera );
    }

    var tableMesh = new THREE.BoxGeometry( 2.74, 0.01, 1.526 );
    var tableMat = new THREE.MeshLambertMaterial( {color: 0x335f51, emissive: 0x335f51, emissiveIntensity: .5} );
    var table = new THREE.Mesh( tableMesh, tableMat );
    table.position.y = -0.01
    table.castShadow= true;
    table.receiveShadow = true;
    scene.add( table );

    function onResize()
    {
      renderer.setSize( window.innerWidth, window.innerHeight );
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
    }

    function animate(time)
    {
        controls.update();
        requestAnimationFrame ( animate );  
        renderer.render (scene, camera);
    }
    animate();
    window.onresize = onResize;
  }

  render() {
    return (
      <div className="viewport"></div>
    )
  }
}
export default Viz