let API_KEY = 'AIzaSyA1MER2l9Ix3nXS6OgPaKTs8fN2mIZmXsU';
let API_VERSION = 'v1';
let CLIENT_ID = '224206679482-o7jif3v8eo0tkfao5d2ilebjf4o4okag.apps.googleusercontent.com';
let SCOPES = 'https://www.googleapis.com/auth/cloud-platform';

function authorize() {
  gapi.client.setApiKey(API_KEY);
  gapi.auth.authorize({
    client_id: CLIENT_ID,
    scope: SCOPES,
    immediate: false
  }, function(authResult) {
       if (authResult && !authResult.error) {
         gapi.client.load('ml', API_VERSION);
       } else {
         window.alert('Auth was not successful');
         console.log(authResult);
       }
     }
  );
};
window.onload = authorize;

function processImage(canvas) {
  ctx = canvas.getContext('2d');
  // scale image to 28x28
  const ctxScaled = document.getElementById('scaled-canvas').getContext('2d')
  ctxScaled.save();
  ctxScaled.clearRect(0, 0, ctxScaled.canvas.height, ctxScaled.canvas.width);
  ctxScaled.scale(28.0 / ctx.canvas.width, 28.0 / ctx.canvas.height)
  ctxScaled.drawImage(document.getElementById('canvas'), 0, 0)
  const {data} = ctxScaled.getImageData(0, 0, 28, 28)
  console.log(data);

  var pixels = []
  var res = 0;
  for (let i = 0, len = data.length; i < len; i += 4) {
    p1 = data[i];
    p2 = data[i+1];
    p3 = data[i+2];
    p4 = data[i+3];
    if (p1 == 0 && p2 == 0 && p3 == 0) {
      if (p4 > 0) {
        res = 1
      } else {
        res = 0
      }
    } else if (p1 == 255 && p2 == 255 && p3 == 255) {
      res = 0
    } else {
      res = 1 - (0.21*p1 + 0.71*p2 + 0.071*p3)/255;
    }
    pixels.push(res)
  }
  ctxScaled.restore();
  //console.log(pixels);
  return pixels;
}

// Canvas setup
var canvas = new fabric.Canvas('canvas');
canvas.isDrawingMode = true;
canvas.freeDrawingBrush.width = 12;
canvas.freeDrawingBrush.color = "#000000";
canvas.backgroundColor = "#ffffff";
canvas.renderAll();

// Clear button callback
$("#clear-canvas").click(function(){ 
  canvas.clear(); 
  canvas.backgroundColor = "#ffffff";
  canvas.renderAll();
  updateChart(zeros);
  $("#status").removeClass();
});

// Predict button callback
$("#predict").click(function(){  

  // Change status indicator
  $("#status").removeClass().toggleClass("fa fa-spinner fa-spin");

  // Send request to Cloud ML Engine prediction service.
  pixels = processImage(canvas);
  console.log(pixels);
  var name = 'projects/mlengine-197219/models/MNIST';
  var request = {
    name: name,
    resource: {'instances': [{'inputs': pixels}]},
  }
  var req = gapi.client.ml.projects.predict(request);
  req.execute(function(resp) {
    console.log(resp);
    $("#status").removeClass().toggleClass("fa fa-check");
    $('#svg-chart').show();
    updateChart(resp.result.predictions[0].probabilities);
  });

/*
      } else {
         $("#status").removeClass().toggleClass("fa fa-exclamation-triangle");
         console.log('Script Error: ' + json.error)
      }
    })
    .fail(function (xhr, textStatus, error) {
      $("#status").removeClass().toggleClass("fa fa-exclamation-triangle");
      console.log("POST Error: " + xhr.responseText + ", " + textStatus + ", " + error);
    }
*/

});

// Iniitialize d3 bar chart
$('#svg-chart').hide();
var labels = ['0','1','2','3','4','5','6','7','8','9'];
var zeros = [0,0,0,0,0,0,0,0,0,0,0];

var margin = {top: 0, right: 0, bottom: 20, left: 0},
    width = 360 - margin.left - margin.right,
    height = 180 - margin.top - margin.bottom;

var svg = d3.select("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", 
          "translate(" + margin.left + "," + margin.top + ")");

var x = d3.scale.ordinal()
    .rangeRoundBands([0, width], .1)
    .domain(labels);
    
var y = d3.scale.linear()
          .range([height, 0])
          .domain([0,1]);  

var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickSize(0);

svg.selectAll(".bar")
    .data(zeros)
  .enter().append("rect")
    .attr("class", "bar")
    .attr("x", function(d, i) { return x(i); })
    .attr("width", x.rangeBand())
    .attr("y", function(d) { return y(d); })
    .attr("height", function(d) { return height - y(d); });

svg.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0," + height + ")")
    .call(xAxis);

// Update chart data
function updateChart(d) {
  d3.selectAll("rect")
    .data(d)
    .transition()
    .duration(500)
    .attr("y", function(d) { return y(d); })
    .attr("height", function(d) { return height - y(d); });
}

var testDigit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3294117748737335, 0.7254902124404907, 0.6235294342041016, 0.5921568870544434, 0.2352941334247589, 0.1411764770746231, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8705883026123047, 0.9960784912109375, 0.9960784912109375, 0.9960784912109375, 0.9960784912109375, 0.9450981020927429, 0.7764706611633301, 0.7764706611633301, 0.7764706611633301, 0.7764706611633301, 0.7764706611633301, 0.7764706611633301, 0.7764706611633301, 0.7764706611633301, 0.6666666865348816, 0.2039215862751007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.26274511218070984, 0.44705885648727417, 0.2823529541492462, 0.44705885648727417, 0.6392157077789307, 0.8901961445808411, 0.9960784912109375, 0.8823530077934265, 0.9960784912109375, 0.9960784912109375, 0.9960784912109375, 0.9803922176361084, 0.8980392813682556, 0.9960784912109375, 0.9960784912109375, 0.5490196347236633, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06666667014360428, 0.25882354378700256, 0.05490196496248245, 0.26274511218070984, 0.26274511218070984, 0.26274511218070984, 0.23137256503105164, 0.08235294371843338, 0.9254902601242065, 0.9960784912109375, 0.41568630933761597, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32549020648002625, 0.9921569228172302, 0.8196079134941101, 0.07058823853731155, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08627451211214066, 0.9137255549430847, 1.0, 0.32549020648002625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5058823823928833, 0.9960784912109375, 0.9333333969116211, 0.1725490242242813, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23137256503105164, 0.9764706492424011, 0.9960784912109375, 0.24313727021217346, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5215686559677124, 0.9960784912109375, 0.7333333492279053, 0.019607843831181526, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03529411926865578, 0.803921639919281, 0.9725490808486938, 0.22745099663734436, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4941176772117615, 0.9960784912109375, 0.7137255072593689, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29411765933036804, 0.9843137860298157, 0.9411765336990356, 0.22352942824363708, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07450980693101883, 0.8666667342185974, 0.9960784912109375, 0.6509804129600525, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.011764707043766975, 0.7960785031318665, 0.9960784912109375, 0.8588235974311829, 0.13725490868091583, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14901961386203766, 0.9960784912109375, 0.9960784912109375, 0.3019607961177826, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12156863510608673, 0.8784314393997192, 0.9960784912109375, 0.45098042488098145, 0.003921568859368563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5215686559677124, 0.9960784912109375, 0.9960784912109375, 0.2039215862751007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2392157018184662, 0.9490196704864502, 0.9960784912109375, 0.9960784912109375, 0.2039215862751007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4745098352432251, 0.9960784912109375, 0.9960784912109375, 0.8588235974311829, 0.1568627506494522, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4745098352432251, 0.9960784912109375, 0.8117647767066956, 0.07058823853731155, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
