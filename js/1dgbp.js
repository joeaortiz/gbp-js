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
    this.factors = [];

    this.sweep_ix = 0;
    this.forward = 1;
    this.past_first_meas = 0;
  }

  update_beliefs() {
    for(var c=0; c<this.var_nodes.length; c++) {
      this.var_nodes[c].update_belief();
    }
  }

  send_messages() {
    for(var c=0; c<this.factors.length; c++) {
      this.factors[c].send_both_mess();
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

    this.factors[adj_var_ids[0]].jacs.push(meas_jac);
    this.factors[adj_var_ids[0]].meas.push(meas);
    this.factors[adj_var_ids[0]].lambdas.push(1 / Math.pow(meas_std, 2));
    this.factors[adj_var_ids[0]].compute_factor();
  }

  computeMAP() {
    var tot_dofs = 0;
    for(var c=0; c<this.var_nodes.length; c++) {
      tot_dofs += this.var_nodes[c].dofs;
    }

    const bigEta = m.Matrix.zeros(tot_dofs, 1);
    const bigLam = m.Matrix.zeros(tot_dofs, tot_dofs);

    for(var c=0; c<this.factors.length; c++) {
      var ix = this.factors[c].adj_var_ids[0];
      bigEta.set(ix, 0, bigEta.get(ix, 0) + this.factors[c].factor.eta.get(0, 0));
      bigEta.set(ix+1, 0, bigEta.get(ix+1, 0) + this.factors[c].factor.eta.get(1, 0));
      bigLam.set(ix, ix, bigLam.get(ix, ix) + this.factors[c].factor.lam.get(0, 0));
      bigLam.set(ix+1, ix, bigLam.get(ix+1, ix) + this.factors[c].factor.lam.get(1, 0));
      bigLam.set(ix, ix+1, bigLam.get(ix, ix+1) + this.factors[c].factor.lam.get(0, 1));
      bigLam.set(ix+1, ix+1, bigLam.get(ix+1, ix+1) + this.factors[c].factor.lam.get(1, 1));
    }


    const bigCov = m.inverse(bigLam);
    const means = bigCov.mmul(bigEta);
    return [means, bigCov];
  }

  compare_to_MAP() {
    var gbp_means = [];
    for(var c=0; c<this.var_nodes.length; c++) {
      gbp_means.push(this.var_nodes[c].belief.getMean().get(0,0));
    }

    const means = new m.Matrix([gbp_means]);
    const map = this.computeMAP()[0];
    var av_diff = (map.sub(means.transpose())).norm();
    return av_diff;
  }

  sweep_step() {
    // Prepare for next step
    var next_ix = 0;
    var next_forward = this.forward;
    if (this.forward) {
      next_ix = this.sweep_ix + 1;
      if (this.sweep_ix == this.var_nodes.length - 2) {
        next_forward = 0;
      } 
    } else {
      next_ix = this.sweep_ix - 1;
      if (this.sweep_ix == 1) {
        next_forward = 1;
      }
    }

    if (this.forward) {
      if (this.factors[this.sweep_ix].meas.length > 1) {
        this.past_first_meas = 1;
      }
      this.factors[this.sweep_ix].send_mess(this.forward);
    } else {
      this.factors[next_ix].send_mess(this.forward);
    }
    if (this.past_first_meas) {
      this.var_nodes[next_ix].update_belief();
    }

    this.sweep_ix = next_ix;
    this.forward = next_forward;
  }
}

class VariableNode {
  constructor(dofs, var_id) {
    this.dofs = dofs;
    this.var_id = var_id;
    this.belief = new Gaussian(m.Matrix.zeros(dofs, 1), m.Matrix.zeros(dofs, dofs));

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
  constructor(dofs, adj_var_ids) {
    this.dofs = dofs;
    this.adj_var_ids = adj_var_ids;
    this.adj_beliefs = [];

    // To compute factor when factor is combination of many factor types (e.g. measurement and smoothness)
    this.jacs = [];
    this.meas = [];
    this.lambdas = [];
    this.factor = new Gaussian(m.Matrix.zeros(dofs, 1), m.Matrix.zeros(dofs, dofs));

    this.messages = [];
  }

  compute_factor() {
    this.factor.eta = m.Matrix.zeros(this.dofs, 1);
    this.factor.lam = m.Matrix.zeros(this.dofs, this.dofs);
    for (var i=0; i<this.jacs.length; i++) {
      this.factor.eta.add(this.jacs[i].transpose().mul(this.lambdas[i] * this.meas[i]));
      this.factor.lam.add(this.jacs[i].transpose().mmul(this.jacs[i]).mul(this.lambdas[i]));
    }
  }

  // Only for bipartite factors where the adjacent vars have 1 dof
  send_mess(ix) {
    if (ix) {
      const mess1 = new Gaussian([[0]], [[0]]);
      mess1.eta = new m.Matrix([[this.factor.eta.get(1, 0) - 
          this.factor.lam.get(1, 0) * (this.factor.eta.get(0, 0) + this.adj_beliefs[0].eta.get(0, 0) - this.messages[0].eta.get(0, 0)) / 
          (this.factor.lam.get(0, 0) + this.adj_beliefs[0].lam.get(0, 0) - this.messages[0].lam.get(0, 0))]]);
      mess1.lam = new m.Matrix([[this.factor.lam.get(1, 1) - 
          this.factor.lam.get(1, 0) * this.factor.lam.get(0, 1) / 
          (this.factor.lam.get(0, 0) + this.adj_beliefs[0].lam.get(0, 0) - this.messages[0].lam.get(0, 0))]]);
      this.messages[1] = mess1;
    } else {
      const mess0 = new Gaussian([[0]], [[0]]);
      mess0.eta = new m.Matrix([[this.factor.eta.get(0, 0) - 
          this.factor.lam.get(0, 1) * (this.factor.eta.get(1, 0) + this.adj_beliefs[1].eta.get(0, 0) - this.messages[1].eta.get(0, 0)) / 
          (this.factor.lam.get(1, 1) + this.adj_beliefs[1].lam.get(0, 0) - this.messages[1].lam.get(0, 0))]]);
      mess0.lam = new m.Matrix([[this.factor.lam.get(0, 0) - 
          this.factor.lam.get(0, 1) * this.factor.lam.get(1, 0) / 
          (this.factor.lam.get(1, 1) + this.adj_beliefs[1].lam.get(0, 0) - this.messages[1].lam.get(0, 0))]]);
      this.messages[0] = mess0;
    }
  }

  send_both_mess(){
    this.send_mess(0);
    this.send_mess(1);
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
    const new_factor = new LinearFactor(2, [i, i+1], );
    new_factor.jacs.push(smoothness_jac);
    new_factor.meas.push(0.);
    new_factor.lambdas.push(1 / Math.pow(smoothness_std, 2));

    new_factor.adj_beliefs.push(graph.var_nodes[i].belief);
    new_factor.adj_beliefs.push(graph.var_nodes[i+1].belief);
    new_factor.messages.push(new Gaussian([[0]], [[0]]));
    new_factor.messages.push(new Gaussian([[0]], [[0]]));
    new_factor.compute_factor();
    graph.factors.push(new_factor);
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

measurements = []

var GBP_on = 0;
var GBP_sweep = 0;
var GBP_sweep_done = 0;
var n_iters = 0;
var dist = 0; // Average distance of belief means from MAP solution
var iters_per_sec = 25;
var disp_MAP = 0;

function syncGBP() {
  if (n_iters == 0 && GBP_sweep_done == 0) {
    // Set initial position of nodes
    for(var c=0; c<graph.var_nodes.length; c++) {
      graph.var_nodes[c].belief.eta = new m.Matrix([[0.05]]);
      graph.var_nodes[c].belief.lam = new m.Matrix([[1e-4]]);
    }
  }
  graph.sync_iter();
  if (!(n_iters == 0)) {
    dist = graph.compare_to_MAP();   
  }
  n_iters++;
}

function drawCanvasBackground() {
  ctx.fillStyle = "grey";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawNodes() {
  for(var c=0; c<graph.var_nodes.length; c++) {
    var x = nodes_x_offset + c*node_x_spacing;
    if (graph.var_nodes[c].belief.lam.get(0, 0) == 0) {
      var y = 500;
      var var_y = Math.pow(50, 2);
    } else {
      var y = graph.var_nodes[c].belief.getMean().get(0, 0);
      var var_y = graph.var_nodes[c].belief.getCov().get(0, 0);
    }

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

function drawMeasurements(meas_model_std) {
  for(var c=0; c<measurements.length; c++) {
    x = measurements[c].x;
    y = measurements[c].y;

    // Draw means
    ctx.beginPath();
    ctx.arc(x, y, node_radius, 0, Math.PI*2);
    // ctx.rect(x - sidelength/2, y - sidelength/2, sidelength, sidelength);
    ctx.fillStyle = "red";
    ctx.fill();
    ctx.closePath();
    // Draw variances
    ctx.beginPath();
    ctx.moveTo(x, parseInt(y) - parseInt(meas_model_std));
    ctx.lineTo(x, parseInt(y) + parseInt(meas_model_std));
    ctx.strokeStyle = "red";
    ctx.stroke();
  }
}

function drawMAP() {
  var values = graph.computeMAP();
  const means = values[0];
  const bigSigma = values[1];
  for(var c=0; c<graph.var_nodes.length; c++) {
    var x = nodes_x_offset + c*node_x_spacing;
    var y = means.get(c, 0);
    var var_y = bigSigma.get(c, c);

    // Draw means
    ctx.beginPath();
    ctx.arc(x, y, node_radius, 0, Math.PI*2);
    ctx.strokeStyle = 'green';
    ctx.stroke();
    // Draw variances
    ctx.beginPath();
    ctx.moveTo(x, parseInt(y) + parseInt(Math.sqrt(var_y)));
    ctx.lineTo(x, parseInt(y) - parseInt(Math.sqrt(var_y)));
    ctx.strokeStyle = 'green';
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
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawCanvasBackground();

  fpsInterval = 1000 / iters_per_sec;
  now = Date.now();
  elapsed = now - then;
  if (elapsed > fpsInterval) {
    then = now - (elapsed % fpsInterval);
    if (GBP_on) {
      syncGBP();
    } else if (GBP_sweep) {
      graph.sweep_step();
      // dist = graph.compare_to_MAP();
      // drawDistance();
      if (graph.forward == 1 && graph.sweep_ix == 0) {
        // Back and forth sweep complete
        GBP_sweep = 0;
        GBP_sweep_done = 1;
      }
    }
  }
  if (GBP_on) {
    drawNumIters();
    drawDistance();
  } 
  drawNodes();
  drawMeasurements(meas_model_std);
  drawDistance();
  if (disp_MAP) {
    drawMAP();
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
    measurements.push({x: relativeX, y: relativeY})
  }
}



// Sliders
function update_meas_model_std(val) {
  meas_model_std = val;
  var lambda = 1 / Math.pow(meas_model_std, 2);
  // document.getElementById("mmstd").innerHTML = meas_model_std;
  for (var c=0; c<graph.factors.length; c++) {
    for (var i=1; i<graph.factors[c].lambdas.length; i++) {
      graph.factors[c].lambdas[i] = lambda;
    }
    graph.factors[c].compute_factor();
  }
}

function update_smoothness_std(val) {
  smoothness_std = val;
  var lambda = 1 / Math.pow(smoothness_std, 2);
  // document.getElementById("sstd").innerHTML = smoothness_std; 
  for (var c=0; c<graph.factors.length; c++) {
    graph.factors[c].lambdas[0] = lambda;
    graph.factors[c].compute_factor();
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
var i_slider = document.getElementById("iters_per_sec");
i_slider.oninput = function() { update_iters_per_sec(this.value)}


// Buttons
function start_sweepGBP() {
  if (measurements.length == 0) {
    alert("You must add a measurement before beginning GBP")
  } else {
    GBP_sweep = 1;
    iters_per_sec = 2;
    GBP_on = 0;
  }
}
function start_syncGBP() {
  if (measurements.length == 0) {
    alert("You must add a measurement before beginning GBP")
  } else {
    if (GBP_on) {
      GBP_on = 0;
      this.value = ("Resume synchronous GBP");
    } else {
      GBP_on = 1;
      GBP_sweep = 0;
      iters_per_sec = i_slider.value;
      this.value = ("Pause synchronous GBP");
    }
  }
}
function display_MAP() {
  if (disp_MAP == 0) {
    disp_MAP = 1;
    this.value = ("Hide MAP");
  } else {
    disp_MAP = 0;
    this.value = ("Display MAP");
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
addButton('map', 'Display MAP', display_MAP);

// On click on canvas
document.addEventListener("click", addMeasurement, false);


startVis(iters_per_sec);
