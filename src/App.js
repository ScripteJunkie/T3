import React, { Component } from "react";
// import ReactDOM from "react-dom";
import * as THREE from "three";
import './App.css'

class App extends Component {
  componentDidMount() {
    // === THREE.JS CODE START ===
    var scene = new THREE.Scene();
    // const canvas = document.querySelector('viewport');
    var camera = new THREE.PerspectiveCamera( 75, window.innerWidth/window.innerHeight, 0.1, 1000 );
    // var renderer = new THREE.WebGLRenderer({ canvas });
    var renderer = new THREE.WebGLRenderer();
    scene.background = new THREE.Color('grey');
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );
    renderer.render( scene, camera );
    var x = -10
    var geometry = new THREE.BoxGeometry( 1, 1, 1 );
    var material = new THREE.MeshBasicMaterial( { color: 0xdadada } );
    for (var i = 0; i < 100; i++)
      window['cube'+i] = new THREE.Mesh( geometry, material );
      eval('cube'+i).position.x = (x+(i/10));
      scene.add( eval('cube'+i) );
      console.log(eval('cube'+i));
      renderer.render( scene, camera );

    // var geometry = new THREE.BoxGeometry( 2, 2, 2 );
    // var material = new THREE.MeshBasicMaterial( { color: 0xdadada } );
    // var cube = new THREE.Mesh( geometry, material );
    // cube.castShadow = true;
    // cube.receiveShadow = true;
    // scene.add( cube );
    // camera.position.z = 5;
    // var animate = function () {
    //   requestAnimationFrame( animate );
    //   cube.rotation.x += 0.01;
    //   cube.rotation.y += 0.01;
    //   cube.rotation.z += 0.01;
    //   renderer.render( scene, camera );
    // };
    // animate();
    // === THREE.JS EXAMPLE CODE END ===
  }
  render() {
    return (
      <div className="viewport"></div>
    )
  }
}
export default App