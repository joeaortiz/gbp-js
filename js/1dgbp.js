var m = require('ml-matrix');


// ****************************** Gaussian ************************************
class Gaussian {
  constructor(eta, lam) {
    // TODO: Consider saving the dimension
    if ((eta instanceof m.Matrix) && (lam instanceof m.Matrix)) {
      this.eta = eta;
      this.lam = lam;
    } else {
      this.eta = new m.Matrix([eta]).transpose();
      this.lam = new m.Matrix(lam);
    }
  }

  getCov() {
    return m.inverse(this.lam);
  }

  getMean() {
    const cov = this.getCov();
    return cov.mmul(this.eta);
  }

  // Take product of gaussian with other gaussian
  product(gaussian) {
    this.eta.add(gaussian.eta);
    this.lam.add(gaussian.lam);
  }


}

// ranges from [i1,i2) & [j1,j2)
function sliceMat(mat,i1,j1,i2,j2) {
  const result = new m.Matrix(i2-i1,j2-j1);
  for(let i = i1; i < i2; ++i) 
    for(let j = j1; j < j2; ++j) {
      result.set(i-i1, j-j1, mat.get(i,j)); 
    }
  return result;
}

// ****************************** GBP ************************************

class FactorGraph {
  constructor() {
    this.var_nodes = [];
    this.smoothness_factors = [];
    this.meas_factors = [];

    this.sweep_ix = 0;
    this.forward = 1;
  }

  update_beliefs() {
    for(var c=0; c<this.var_nodes.length; c++) {
      this.var_nodes[c].update_belief();
    }
  }

  send_messages() {
    for(var c=0; c<this.smoothness_factors.length; c++) {
      this.smoothness_factors[c].send_mess();
    }
    for(var c=0; c<this.meas_factors.length; c++) {
      this.meas_factors[c].send_mess();
    }
  }

  sync_iter() {
    this.send_messages();
    this.update_beliefs();
  }

  addLinearMeasurement(meas, x_meas, adj_var_ids,
                    x_var_lhs, x_var_rhs, meas_std) {
    var gamma = (x_meas - x_var_lhs) / (x_var_rhs - x_var_lhs);
    const meas_jac = new m.Matrix([[1 - gamma, gamma]]);

    const new_factor = new LinearFactor(meas, meas_jac, 2, adj_var_ids, meas_std, "measurement", x_meas);
    new_factor.adj_beliefs.push(this.var_nodes[adj_var_ids[0]].belief);
    new_factor.adj_beliefs.push(this.var_nodes[adj_var_ids[1]].belief);
    new_factor.messages.push(new Gaussian([[0]], [[0]]));
    new_factor.messages.push(new Gaussian([[0]], [[0]]));
    this.meas_factors.push(new_factor);
    this.var_nodes[adj_var_ids[0]].adj_factors.push(new_factor)
    this.var_nodes[adj_var_ids[1]].adj_factors.push(new_factor)
  }

  computeMAP() {
    var tot_dofs = 0;
    for(var c=0; c<this.var_nodes.length; c++) {
      tot_dofs += this.var_nodes[c].dofs;
    }

    const bigEta = m.Matrix.zeros(tot_dofs, 1);
    const bigLam = m.Matrix.zeros(tot_dofs, tot_dofs);

    for(var c=0; c<this.smoothness_factors.length; c++) {
      var ix = this.smoothness_factors[c].adj_var_ids[0];
      bigEta.set(ix, 0, bigEta.get(ix, 0) + this.smoothness_factors[c].factor.eta.get(0, 0));
      bigEta.set(ix+1, 0, bigEta.get(ix+1, 0) + this.smoothness_factors[c].factor.eta.get(1, 0));
      bigLam.set(ix, ix, bigLam.get(ix, ix) + this.smoothness_factors[c].factor.lam.get(0, 0));
      bigLam.set(ix+1, ix, bigLam.get(ix+1, ix) + this.smoothness_factors[c].factor.lam.get(1, 0));
      bigLam.set(ix, ix+1, bigLam.get(ix, ix+1) + this.smoothness_factors[c].factor.lam.get(0, 1));
      bigLam.set(ix+1, ix+1, bigLam.get(ix+1, ix+1) + this.smoothness_factors[c].factor.lam.get(1, 1));
    }
    for(var c=0; c<this.meas_factors.length; c++) {
      var ix = this.meas_factors[c].adj_var_ids[0];
      bigEta.set(ix, 0, bigEta.get(ix, 0) + this.meas_factors[c].factor.eta.get(0, 0));
      bigEta.set(ix+1, 0, bigEta.get(ix+1, 0) + this.meas_factors[c].factor.eta.get(1, 0));
      bigLam.set(ix, ix, bigLam.get(ix, ix) + this.meas_factors[c].factor.lam.get(0, 0));
      bigLam.set(ix+1, ix, bigLam.get(ix+1, ix) + this.meas_factors[c].factor.lam.get(1, 0));
      bigLam.set(ix, ix+1, bigLam.get(ix, ix+1) + this.meas_factors[c].factor.lam.get(0, 1));
      bigLam.set(ix+1, ix+1, bigLam.get(ix+1, ix+1) + this.meas_factors[c].factor.lam.get(1, 1));
    }

    const bigCov = m.inverse(bigLam);
    const means = bigCov.mmul(bigEta);
    return means;
  }

  compare_to_MAP() {
    var gbp_means = [];
    for(var c=0; c<this.var_nodes.length; c++) {
      gbp_means.push(this.var_nodes[c].belief.getMean().get(0,0));
    }

    const means = new m.Matrix([gbp_means]);
    const map = this.computeMAP();
    var av_diff = (map.sub(means.transpose())).norm();
    return av_diff;
  }

  sweep_step() {
    // Prepare for next step
    var next_ix = 0;
    if (this.forward) {
      if (this.sweep_ix == this.var_nodes.length -1) {
        next_ix = this.sweep_ix - 1;
      } else {
        next_ix = this.sweep_ix + 1;
      }
    } else {
      if (this.sweep_ix == 0) {
        next_ix = this.sweep_ix + 1;
      } else {
        next_ix = this.sweep_ix - 1;
      }
    }

    this.var_nodes[this.sweep_ix].update_belief();
    // Connected factors which also link to the var node on the right send messages
    for (var i=0; i<this.var_nodes[this.sweep_ix].adj_factors.length; i++) {
      if (this.var_nodes[this.sweep_ix].adj_factors[i].adj_var_ids[1] == next_ix) {
        this.var_nodes[this.sweep_ix].adj_factors[i].send_mess();
      }
    }
    this.sweep_ix = next_ix;
  }
}

class VariableNode {
  constructor(dofs, var_id) {
    this.dofs = dofs;
    this.var_id = var_id;
    this.belief = new Gaussian(m.Matrix.zeros(dofs, 1), m.Matrix.eye(dofs, dofs).mul(1e-7));

    this.adj_factors = [];
  }

  update_belief() {
    this.belief.eta = m.Matrix.zeros(this.dofs, 1);
    this.belief.lam = m.Matrix.zeros(this.dofs, this.dofs);

    // Take product of incoming messages
    for(var c=0; c<this.adj_factors.length; c++) {
      var ix = this.adj_factors[c].adj_var_ids.indexOf(this.var_id);
      this.belief.product(this.adj_factors[c].messages[ix])
    }

    // Send new belief to adjacent factors
    for(var c=0; c<this.adj_factors.length; c++) {
      var ix = this.adj_factors[c].adj_var_ids.indexOf(this.var_id);
      this.adj_factors[c].adj_beliefs[ix] = this.belief;
    }
  }
}


class LinearFactor {
  constructor(meas, jac, dofs, adj_var_ids, meas_model_std, type, x_meas) {
    this.type = type;
    this.x_meas = x_meas;
    this.jac = jac;
    this.adj_var_ids = adj_var_ids;
    this.adj_beliefs = [];
    this.messages = [];

    this.meas = meas;
    var lambda = 1 / Math.pow(meas_model_std, 2);
    this.factor = new Gaussian(jac.transpose().mul(lambda * this.meas), 
                                     jac.transpose().mmul(jac).mul(lambda));

  }

  // Only for bipartite factors where the adjacent vars have 1 dof
  send_mess() {
    const mess0 = new Gaussian([[0]], [[0]]);
    const mess1 = new Gaussian([[0]], [[0]]);

    mess0.eta = new m.Matrix([[this.factor.eta.get(0, 0) - 
        this.factor.lam.get(0, 1) * (this.factor.eta.get(1, 0) + this.adj_beliefs[1].eta.get(0, 0) - this.messages[1].eta.get(0, 0)) / 
        (this.factor.lam.get(1, 1) + this.adj_beliefs[1].lam.get(0, 0) - this.messages[1].lam.get(0, 0))]]);
    mess0.lam = new m.Matrix([[this.factor.lam.get(0, 0) - 
        this.factor.lam.get(0, 1) * this.factor.lam.get(1, 0) / 
        (this.factor.lam.get(1, 1) + this.adj_beliefs[1].lam.get(0, 0) - this.messages[1].lam.get(0, 0))]]);

    mess1.eta = new m.Matrix([[this.factor.eta.get(1, 0) - 
        this.factor.lam.get(1, 0) * (this.factor.eta.get(0, 0) + this.adj_beliefs[0].eta.get(0, 0) - this.messages[0].eta.get(0, 0)) / 
        (this.factor.lam.get(0, 0) + this.adj_beliefs[0].lam.get(0, 0) - this.messages[0].lam.get(0, 0))]]);
    mess1.lam = new m.Matrix([[this.factor.lam.get(1, 1) - 
        this.factor.lam.get(1, 0) * this.factor.lam.get(0, 1) / 
        (this.factor.lam.get(0, 0) + this.adj_beliefs[0].lam.get(0, 0) - this.messages[0].lam.get(0, 0))]]);

    this.messages[0] = mess0;
    this.messages[1] = mess1;
  }
}


function create1Dgraph(n_var_nodes, smoothness_std) {

  const graph = new FactorGraph()

  // Create variable nodes
  for(var i=0; i<n_var_nodes; i++) {
    const new_var_node = new VariableNode(1, i);
    graph.var_nodes.push(new_var_node);
  }

  // Create smoothness factors
  const smoothness_jac = new m.Matrix([[-1, 1]]);
  for(var i=0; i<(n_var_nodes-1); i++) {
    const new_factor = new LinearFactor(0., smoothness_jac, 2, [i, i+1], smoothness_std, "smoothness" , "");
    new_factor.adj_beliefs.push(graph.var_nodes[i].belief);
    new_factor.adj_beliefs.push(graph.var_nodes[i+1].belief);
    new_factor.messages.push(new Gaussian([[0]], [[0]]));
    new_factor.messages.push(new Gaussian([[0]], [[0]]));
    graph.smoothness_factors.push(new_factor);
    graph.var_nodes[i].adj_factors.push(new_factor);
    graph.var_nodes[i+1].adj_factors.push(new_factor);
  }

  return graph;
}


// ****************************** Run GBP ************************************

// Visual varaibles
var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
ctx.lineWidth = 3

var node_radius = 10;
var sidelength = 25;

// GBP variables
var n_var_nodes = 10;
var nodes_x_offset = 50;
var node_x_spacing = (canvas.width - 100) / (n_var_nodes - 1)

var meas_model_std = 50;
var smoothness_std = 50;

const graph = create1Dgraph(10, 50);
// Set initial position of nodes
for(var c=0; c<graph.var_nodes.length; c++) {
  graph.var_nodes[c].belief.eta = new m.Matrix([[0.05]]);
  graph.var_nodes[c].belief.lam = new m.Matrix([[1e-4]]);
}

var GBP_on = 0;
var GBP_sweep = 0;
var n_iters = 0;
var dist = 0; // Average distance of belief means from MAP solution
var iters_per_sec = 25;

function syncGBP() {
  graph.sync_iter();
  dist = graph.compare_to_MAP();
  n_iters++;
}

function drawCanvasBackground() {
  ctx.fillStyle = "grey";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawNodes() {
  for(var c=0; c<graph.var_nodes.length; c++) {
    var x = nodes_x_offset + c*node_x_spacing;
    var y = graph.var_nodes[c].belief.getMean().get(0, 0);
    var var_y = graph.var_nodes[c].belief.getCov().get(0, 0);

    // Draw means
    ctx.beginPath();
    ctx.arc(x, y, node_radius, 0, Math.PI*2);
    ctx.fillStyle = "#0095DD";
    ctx.fill();
    ctx.closePath();
    // Draw variances
    ctx.beginPath();
    ctx.moveTo(x, parseInt(y) + parseInt(Math.sqrt(var_y)));
    ctx.lineTo(x, parseInt(y) - parseInt(Math.sqrt(var_y)));
    ctx.strokeStyle = "#0095DD";
    ctx.stroke();

  }
}

function drawMeasurements() {
  for(var c=0; c<graph.meas_factors.length; c++) {
    x_meas = graph.meas_factors[c].x_meas;
    y = graph.meas_factors[c].meas;

    // Draw means
    ctx.beginPath();
    ctx.arc(x_meas, y, node_radius, 0, Math.PI*2);
    // ctx.rect(x_meas - sidelength/2, y - sidelength/2, sidelength, sidelength);
    ctx.fillStyle = "red";
    ctx.fill();
    ctx.closePath();
    // Draw variances
    ctx.beginPath();
    ctx.moveTo(x_meas, parseInt(y) - parseInt(meas_model_std));
    ctx.lineTo(x_meas, parseInt(y) + parseInt(meas_model_std));
    ctx.strokeStyle = "red";
    ctx.stroke();

  }
}

function drawDistance() {
    ctx.font = "16px Arial";
    ctx.fillStyle = "black";
    ctx.fillText("Av dist from MAP: "+dist.toFixed(4), 8, 20);
}

function drawNumIters() {
    ctx.font = "16px Arial";
    ctx.fillStyle = "black";
    ctx.fillText("Num iterations: "+n_iters, canvas.width - 170, 20);
}


function startVis(fps) {
    then = Date.now();
    startTime = then;
    updateVis();
}

function updateVis() {
  requestAnimationFrame(updateVis);

  fpsInterval = 1000 / iters_per_sec;
  now = Date.now();
  elapsed = now - then;
  if (elapsed > fpsInterval) {
    then = now - (elapsed % fpsInterval);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawCanvasBackground();
    if (GBP_on) {
      syncGBP();
      drawNumIters();
      drawDistance();
    } else if (GBP_sweep) {
      graph.sweep_step();
      dist = graph.compare_to_MAP();
      drawDistance();
    }
    drawNodes();
    drawMeasurements();
    drawDistance();
  }
}

function addMeasurement(e) {
  var relativeX = e.clientX - canvas.offsetLeft;
  var relativeY = e.clientY - canvas.offsetTop;
  if(relativeX > 0 && relativeX < canvas.width && relativeY > 0 && relativeY < canvas.height) {
    var ix = (relativeX - nodes_x_offset) / node_x_spacing;
    var x_lhs = nodes_x_offset + Math.floor(ix)*node_x_spacing;
    var x_rhs = nodes_x_offset + Math.ceil(ix)*node_x_spacing;
    graph.addLinearMeasurement(relativeY, relativeX, [Math.floor(ix), Math.ceil(ix)], x_lhs, x_rhs, meas_model_std);
  }
}


// Sliders
function update_meas_model_std(val) {
  meas_model_std = val;
  // document.getElementById("mmstd").innerHTML = meas_model_std;
  var lambda = 1 / Math.pow(meas_model_std, 2);

  for(var c=0; c<graph.meas_factors.length; c++) {
    graph.meas_factors[c].factor.eta = graph.meas_factors[c].jac.transpose().mul(lambda * graph.meas_factors[c].meas)
    graph.meas_factors[c].factor.lam = graph.meas_factors[c].jac.transpose().mmul(graph.meas_factors[c].jac).mul(lambda);
  }
}

function update_smoothness_std(val) {
  smoothness_std = val;
  // document.getElementById("sstd").innerHTML = smoothness_std; 
  var lambda = 1 / Math.pow(smoothness_std, 2);

  for(var c=0; c<graph.smoothness_factors.length; c++) {
    graph.smoothness_factors[c].factor.eta = graph.smoothness_factors[c].jac.transpose().mul(lambda * graph.smoothness_factors[c].meas)
    graph.smoothness_factors[c].factor.lam = graph.smoothness_factors[c].jac.transpose().mmul(graph.smoothness_factors[c].jac).mul(lambda);
  }
}

function update_iters_per_sec(val) {
  iters_per_sec = val;
  document.getElementById("fps").innerHTML = iters_per_sec; 
}

var mm_slider = document.getElementById("meas_model_std");
mm_slider.oninput = function() { update_meas_model_std(this.value)}
var s_slider = document.getElementById("smoothness_std");
s_slider.oninput = function() { update_smoothness_std(this.value)}
var s_slider = document.getElementById("iters_per_sec");
s_slider.oninput = function() { update_iters_per_sec(this.value)}


// Buttons
function start_sweepGBP() {
  alert("To do")
  // if (graph.meas_factors.length == 0) {
  //   alert("You must add a measurement before beginning GBP")
  // } else {
  //   GBP_sweep = 1;
  //   iters_per_sec = 2;
  // }
}
function start_syncGBP() {
  if (graph.meas_factors.length == 0) {
    alert("You must add a measurement before beginning GBP")
  } else {
    if (GBP_on) {
      GBP_on = 0;
      this.value = ("Resume synchronous GBP");
    } else {
      GBP_on = 1;
      this.value = ("Pause synchronous GBP");
    }
  }
}

function addButton(name, text, func) {
  var buttonnode = document.createElement('input');
  buttonnode.setAttribute('type','button');
  buttonnode.setAttribute('value', text);
  buttonnode.addEventListener ("click", func, false);
  document.getElementById(name).appendChild(buttonnode);
}

addButton('sync', 'Start synchronous GBP', start_syncGBP);
addButton('sweep', 'Start a sweep of GBP', start_sweepGBP);

// On click on canvas
document.addEventListener("click", addMeasurement, false);


startVis(iters_per_sec);
