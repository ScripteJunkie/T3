var src1 = document.createElement("img");
var src2 = document.createElement("img");
src1.src = "http://129.21.124.68:8091/";
src2.src = "http://129.21.124.68:8092/";

var cam1 = document.getElementById("cam1");
var cam2 = document.getElementById("cam2");
src1.width = cam1.offsetWidth;
src1.height = cam1.offsetHeight;
src2.width = cam2.offsetWidth;
src2.height = cam2.offsetHeight;
cam1.appendChild(src1);
cam2.appendChild(src2);