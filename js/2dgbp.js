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

  getCovEllipse() {
    const cov = this.getCov();
    var e = new m.EigenvalueDecomposition(cov);
    var real = e.realEigenvalues;

    // Eigenvectors are always orthogonal
    var vectors = e.eigenvectorMatrix;
    var angle = Math.atan(vectors.get(1, 0) / vectors.get(0, 0))
    return [real, angle]
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

}

class VariableNode {
  constructor(dofs, var_id) {
    this.dofs = dofs;
    this.var_id = var_id;
    this.belief = new Gaussian(m.Matrix.zeros(dofs, 1), m.Matrix.zeros(dofs, dofs));
    this.prior = new Gaussian(m.Matrix.zeros(dofs, 1), m.Matrix.zeros(dofs, dofs));

    this.adj_factors = [];
  }

  update_belief() {
    this.belief.eta = this.prior.eta;
    this.belief.lam = this.prior.lam;

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


function create2Dgraph(n_var_nodes, smoothness_std) {

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

var robot_loc = [50, 590];
var last_key_pose = [50, 590];
var step = 10;
var new_pose_dist = 70;

var landmarks = [];

var GBP_on = 0;
var n_iters = 0;

const graph = new FactorGraph();
// Create variable node at starting position
const first_var_node = new VariableNode(2, 0);
first_var_node.prior.eta = new m.Matrix([[robot_loc[0]], [robot_loc[1]]]);
first_var_node.prior.lam = new m.Matrix([[1, 0], [0, 1]]);
first_var_node.update_belief();
graph.var_nodes.push(first_var_node);






// Visual varaibles
var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
ctx.lineWidth = 3;

var node_radius = 10;
var lmk_radius = 6;

// GBP variables
var n_var_nodes = 2;

var meas_model_std = 50;
var smoothness_std = 50;


var var_nodes = []
// Create nodes
const node1 = new Gaussian(new m.Matrix([[0.1], [0.1]]), new m.Matrix([[0.001, 0], [0, 0.001]]));
var_nodes.push(node1);
const node2 = new Gaussian(new m.Matrix([[0.4], [0.3]]), new m.Matrix([[0.001, 0], [0, 0.001]]));
var_nodes.push(node2);



function drawCanvasBackground() {
  ctx.fillStyle = "grey";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawRobot() {
  ctx.beginPath();
  ctx.arc(robot_loc[0], robot_loc[1], node_radius, 0, Math.PI*2);
  ctx.fillStyle = "green";
  ctx.fill();
  ctx.closePath();
}

function drawNodes() {
  for(var c=0; c<graph.var_nodes.length; c++) {
    const mean = graph.var_nodes[c].belief.getMean();
    var x = mean.get(0, 0);
    var y = mean.get(1, 0);

    // Draw means
    ctx.beginPath();
    ctx.arc(x, y, node_radius, 0, Math.PI*2);
    ctx.fillStyle = "#0095DD";
    ctx.fill();
    ctx.closePath();

    var values = graph.var_nodes[c].belief.getCovEllipse();
    var eig_values = values[0];
    var angle = values[1];

    // Draw variances
    ctx.beginPath();
    ctx.ellipse(x, y, Math.sqrt(eig_values[0]), Math.sqrt(eig_values[1]), angle, 0, 2*Math.PI)
    ctx.strokeStyle = "#0095DD";
    ctx.stroke();
  }
}

function drawLandmarks() {
  for(var i=0; i<landmarks.length; i++) {
    ctx.beginPath();
    ctx.arc(landmarks[i].x, landmarks[i].y, lmk_radius, 0, Math.PI*2);
    ctx.fillStyle = "orange";
    ctx.fill();
    ctx.closePath();
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


function updateVis() {
  requestAnimationFrame(updateVis);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  drawCanvasBackground();

  // drawNumIters();
  // drawDistance();
  drawLandmarks();
  drawRobot();
  drawNodes();
}

function startVis(fps) {
    // then = Date.now();
    // startTime = then;
    updateVis();
}

document.addEventListener("keydown", checkKey);
document.addEventListener("click", addLandmark, false);


function checkKey(e) {
  console.log(canvas.width, ctx.height)
  e = e || window.event;
  if (e.keyCode == '38') {
    if (robot_loc[1] > node_radius + step) {
      robot_loc[1] -= step;
    }
  }
  else if (e.keyCode == '40') {
    if (robot_loc[1] < canvas.height - node_radius - step) {
      robot_loc[1] += step;
    }
  }
  else if (e.keyCode == '37') {
    if (robot_loc[0] > node_radius + step) {
      robot_loc[0] -= step;
    }
  }
  else if (e.keyCode == '39') {
    if (robot_loc[0] < canvas.width - node_radius - step) {
      robot_loc[0] += step;
    }
  }
  checkAddVarNode();
}

function addLandmark(e) {
  var relativeX = e.clientX - canvas.offsetLeft;
  var relativeY = e.clientY - canvas.offsetTop;
  if(relativeX > 0 && relativeX < canvas.width && relativeY > 0 && relativeY < canvas.height) {
    landmarks.push({x: relativeX, y: relativeY});
  }
}

function checkAddVarNode() {
  var dist = Math.sqrt(Math.pow(robot_loc[0] - last_key_pose[0], 2) + 
                       Math.pow(robot_loc[1] - last_key_pose[1], 2));

  if (dist > new_pose_dist) {
    const new_var_node = new VariableNode(2, graph.var_nodes.length);
    new_var_node.prior.lam = new m.Matrix([[0.0004, 0], [0, 0.0004]]);
    new_var_node.prior.eta = new_var_node.prior.lam.mmul(new m.Matrix([[robot_loc[0]], [robot_loc[1]]]))
    new_var_node.update_belief()
    graph.var_nodes.push(new_var_node);
    last_key_pose[0] = robot_loc[0];
    last_key_pose[1] = robot_loc[1];
  }
}


startVis();
