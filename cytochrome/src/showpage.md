# Highlight 



- Smooth Scroll Skeleton using Lenis + gsap 

- The page should use all three models where necessary, the idea is, we can never run out of 3d material for the page


- The proposed page is supposed to be "fireworks" and I know exactly how to do that, and I need you to try your best to implement it. 

I have added new code files, in new folders 
cytochrome/src/helpers
cytochrome/src/templates
cytochrome/components/canvas 
cytochrome/components/dom

So, the first idea is, exploding a glb and then showing specific parts controlled by gsap scrolltrigger 

`<div className="h-full text-[22px] text-[white] m-0">
  <div className="heading-wrapper">
    <h1 className="red stack heading-6">
      Scrollytelling with GSAP ScrollTrigger
    </h1>
    <h1 className="orange stack heading-5">
      Scrollytelling with GSAP ScrollTrigger
    </h1>
    <h1 className="yellow stack heading-4">
      Scrollytelling with GSAP ScrollTrigger
    </h1>
    <h1 className="green stack heading-3">
      Scrollytelling with GSAP ScrollTrigger
    </h1>
    <h1 className="blue stack heading-2">
      Scrollytelling with GSAP ScrollTrigger
    </h1>
    <h1 className="purple stack heading-1">
      Scrollytelling with GSAP ScrollTrigger
    </h1>
  </div>
  <p>
    GSAP is a JavaScript library that makes it easy to code high-performance and
    complex animations. GSAP has a flexible interface that is easy to use with
    D3 and other common dataviz libraries. GSAP has just released a new plugin
    called ScrollTrigger that facilitates scroll-driven animations.
    ScrollTrigger can be used along with GSAP’s own animation functions, but you
    can also use it just as a scroll watcher to trigger any function (for
    example, run some D3 code) on a particular scroll interaction. This document
    showcases how you can use ScrollTrigger to power some common dataviz
    scrollytelling patterns, but the library is extremely full-featured and
    flexible, you can see more about what’s possible
    <a href="https://greensock.com/scrolltrigger" target="_blank">
      in the docs
    </a>
    . Oh, and if you want to see all the scroller and trigger markers for each
    scroll interaction in this demo, use the menu at the top right ↗.
  </p>
  <p>
    <a href="https://github.com/will-r-chase/reforma/tree/master/content/scroll_trigger_demo">
      See the source code for this demo!
    </a>
  </p>
  <p>
    <i>
      *Disclaimer: ScrollTrigger is very new, and hence I have had very little
      practice with it and I doubt it has even been used by anyone in a real
      dataviz project yet. That means there’s very few best practices at this
      point. What follows works for me, but it is subject to change, and I make
      no claims that this is the best or “correct” way to do things.
    </i>
  </p>
  <hr />
  <div id="chart-wrapper" className='w-3/5 border ml-[40%] border-solid border-[black]'>
    <svg />
  </div>
  <article id="scroll-steps">
    <section className="step" id="step-1">
      One common scrollytelling pattern is to have a chart become ‘pinned’,
      while boxes of explainer text are scrolled, like this. GSAP has a built-in
      method for pinning elements like this. This chart is pinned by creating a
      new <span className="code">ScrollTrigger</span> which triggers when the{" "}
      <span className="code">#chart-wrapper</span> element reaches the middle of
      the viewport. The trigger sets
      <span className="code">pin: true</span> to pin the trigger element (in
      this case our chart wrapper) in place until the end of the trigger, which
      is set to the bottom of the last text box.
      <pre>
        {"            "}
        <code>
          {"\n"}ScrollTrigger.create({"{"}
          {"\n"}
          {"    "}trigger: '#chart-wrapper',{"\n"}
          {"    "}endTrigger: '#step-4',{"\n"}
          {"    "}start: 'center center',{"\n"}
          {"    "}end: () =&gt; {"{"}
          {"\n"}
          {"        "}const height = window.innerHeight;{"\n"}
          {"        "}const chartHeight =
          document.querySelector('#chart-wrapper').offsetHeight;{"\n"}
          {"        "}return `bottom ${"{"}chartHeight + (height - chartHeight)
          / 2{"}"}px`;{"\n"}
          {"    "}
          {"}"},{"\n"}
          {"    "}pin: true,{"\n"}
          {"    "}pinSpacing: false{"\n"}
          {"}"});{"\n"}
          {"            "}
        </code>
        {"\n"}
        {"        "}
      </pre>
    </section>
    <section className="step" id="step-2">
      You can then trigger GSAP animations. Here, we’re using a normal
      <span className="code">gsap.to()</span> tween to animate the
      <span className="code">cx</span> attribute of our points. To trigger the
      animation on scroll, we just add a
      <span className="code">scrollTrigger</span> object to our normal GSAP
      tween. The animation fires when the second text box (our
      <span className="code">trigger</span> ) crosses the middle of the screen,
      and reverses when you scroll backwards. Using GSAP tweens has the
      advantage that you can easily pause, restart, complete, or reverse your
      tweens using <span className="code">toggleActions</span>.
      <pre>
        {"            "}
        <code>
          {"\n"}gsap.to('.points', {"{"}
          {"\n"}
          {"    "}scrollTrigger: {"{"}
          {"\n"}
          {"        "}trigger: '#step-2',{"\n"}
          {"        "}start: 'top center',{"\n"}
          {"        "}toggleActions: 'play none none reverse'{"\n"}
          {"    "}
          {"}"},{"\n"}
          {"    "}cx: () =&gt; Math.random() * svgWidth,{"\n"}
          {"    "}duration: 0.5,{"\n"}
          {"    "}ease: 'power3.inOut'{"\n"}
          {"}"});{"\n"}
          {"            "}
        </code>
        {"\n"}
        {"        "}
      </pre>
    </section>
    <section className="step" id="step-3">
      You can also just use GSAP to set the ScrollTrigger, and let D3 or another
      library handle the animation. In this case, we set up a new
      <span className="code">ScrollTrigger</span> with GSAP, and passed it
      callback functions (<span className="code">circlesToTimeline</span> and
      <span className="code">circlesToRandom</span>) which contain our custom D3
      animation code via the arguments
      <span className="code">onEnter</span> (forward animation) and
      <span className="code">onLeaveBack</span> onLeaveBack (backwards
      animation). The <span className="code">onEnter</span> callback will fire
      when our
      <span className="code">trigger</span> crosses the scroller start from the
      top, and onLeaveBack will fire when the
      <span className="code">trigger</span> crosses the scroller start from the
      bottom.
      <pre>
        {"            "}
        <code>
          {"\n"}ScrollTrigger.create({"{"}
          {"\n"}
          {"    "}trigger: '#step-3',{"\n"}
          {"    "}start: 'top center',{"\n"}
          {"    "}onEnter: circlesToTimeline,{"\n"}
          {"    "}onLeaveBack: circlesToRandom{"\n"}
          {"}"});{"\n"}
          {"            "}
        </code>
        {"\n"}
        {"        "}
      </pre>
    </section>
    <section className="step" id="step-4">
      Also, notice how all these text boxes are transitioning from transparent
      to opaque when they enter and leave the viewport? That’s GSAP too. We set
      up a <span className="code">ScrollTrigger</span> for each box (it was easy
      with GSAP’s <span className="code">toArray</span> function and a
      <span className="code">forEach</span> loop), and used the
      <span className="code">toggleClass</span> argument to toggle an ‘active’
      CSS class (which just adds opacity to the element) whenever the element is
      in view.
      <pre>
        {"            "}
        <code>
          {"\n"}gsap.utils.toArray('.step').forEach(step =&gt; {"{"}
          {"\n"}
          {"    "}ScrollTrigger.create({"{"}
          {"\n"}
          {"        "}trigger: step,{"\n"}
          {"        "}start: 'top 80%',{"\n"}
          {"        "}end: 'center top',{"\n"}
          {"        "}toggleClass: 'active'{"\n"}
          {"    "}
          {"}"});{"\n"}
          {"}"});{"\n"}
          {"            "}
        </code>
        {"\n"}
        {"        "}
      </pre>
    </section>
  </article>
  <p>
    So far we’ve been animating SVG elements, but GSAP can animate components of
    basically any renderer: SVG, DOM, Canvas, or WebGL. And it’s built to work
    with popular libraries like pixi.js and three.js. So, let’s try using it to
    animate some DOM elements, in this case a bunch of stacked text blocks.
    We’ll also show off another cool features of ScrollTrigger—scrubbing. In the
    examples above, the whole animation fired as soon as the trigger was
    activated, but you can also link the animation progress to the scroll
    progress, so that when you’re scrolled 50% into an element, the animation is
    50% complete. In GSAP, this is as easy as setting{" "}
    <span className="code">scrub: true</span> on your
    <span className="code">scrollTrigger</span>. You can also set scrub to a
    number, which will basically put a delay on the animation so that it catches
    up to the scroll position that many seconds later. Here we’ve stacked six
    text blocks, each colored differently, and set them to move down the page
    with a scrubbing animation. Each text block has a slight scrub delay (the
    purple text has almost no delay, and the red text has the longest delay), to
    demonstrate the ‘catch-up’ feature. The code looks like this:
  </p>
  <pre className="text-centered">
    {"            "}
    <code>
      {"\n"}gsap.utils.toArray('.scrub').forEach((el, i) =&gt; {"{"}
      {"\n"}
      {"    "}gsap.to(el, {"{"}
      {"\n"}
      {"        "}scrollTrigger: {"{"}
      {"\n"}
      {"            "}trigger: '.scrub-wrapper',{"\n"}
      {"            "}start: 'top top',{"\n"}
      {"            "}end: 'bottom center+=150',{"\n"}
      {"            "}pin: '.scrub-wrapper',{"\n"}
      {"            "}scrub: (7 - i) * 0.1{"\n"}
      {"        "}
      {"}"},{"\n"}
      {"        "}y: '45vh'{"\n"}
      {"    "}
      {"}"});{"\n"}
      {"}"});{"\n"}
      {"            "}
    </code>
    {"\n"}
    {"        "}
  </pre>
  <p />
  <div className="scrub-wrapper">
    <h1 className="red stack scrub">scroll me!</h1>
    <h1 className="orange stack scrub">scroll me!</h1>
    <h1 className="yellow stack scrub">scroll me!</h1>
    <h1 className="green stack scrub">scroll me!</h1>
    <h1 className="blue stack scrub">scroll me!</h1>
    <h1 className="purple stack scrub">scroll me!</h1>
  </div>
  <p>
    This has barely scratched the surface of what’s possible with GSAP and
    ScrollTrigger, but hopefully you can see how powerful this library is for
    dataviz and scrollytelling. Although this post has emphasized how simple
    ScrollTrigger makes all of this, the truth is that there’s always gotcha’s
    that come up with scrollytelling, it’s usually not that simple. Making sure
    that your CSS and HTML structure don’t fight with the scroll interactions
    can take some practice and trial and error. I’ve also found that GSAP
    animations work quite differently than D3 animations, so it can be tricky to
    mix these two. ScrollTrigger also has quite a bewildering array of options
    for configuration. This makes it far more flexible and powerful than most
    other scrollytelling libraries, but it also means that setting up animations
    can take some getting used to.
  </p>
  <p>
    This demo is just to show off what’s possible and give some code examples,
    but notice I didn’t talk much about the details, like setting triggers or
    start and end points. You can read more about the details, other options,
    and helpful tips for ScrollTrigger{" "}
    <a href="https://www.williamrchase.com/post/scrollytelling-with-gsap-scrolltrigger/">
      on my blog
    </a>
    . You can also see the full source code for this demo{" "}
    <a href="https://github.com/will-r-chase/reforma/tree/master/content/scroll_trigger_demo">
      on my GitHub
    </a>
    .
  </p>
</>
`

and here is the vanilla js code 
`gsap.registerPlugin(ScrollTrigger);
const purple = '#854794';
const blue = '#00A8DE';
const green = '#54AE37';
const yellow = '#FFDB00';
const orange = '#F5A336';
const red = '#E84750';
const rainbow = [red, orange, yellow, green, blue, purple];

//////////////////////////////////////////////////
// This is all just for the menu, nothing to do //
// with scrolling or animations                 //
//////////////////////////////////////////////////
const pinCheck = document.querySelector('#pin');
const toggleCheck = document.querySelector('#toggle');
const box1Check = document.querySelector('#box1');
const box2Check = document.querySelector('#box2');
const box3Check = document.querySelector('#box3');
const scrubCheck = document.querySelector('#scrub');
const pinLabel = document.querySelector('#pin-label');
const toggleLabel = document.querySelector('#toggle-label');
const box1Label = document.querySelector('#box1-label');
const box2Label = document.querySelector('#box2-label');
const box3Label = document.querySelector('#box3-label');
const scrubLabel = document.querySelector('#scrub-label');
const checkLabels = document.querySelectorAll('.check-label');
let menuCollapsed = true;
const menu = document.querySelector('.checkbox-group');

function contains(selector, text) {
  var elements = document.querySelectorAll(selector);
  return Array.prototype.filter.call(elements, function(element) {
    return RegExp(text).test(element.textContent);
  });
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////

//loads and then cleans some data I used previously in a different project
//basically just a set of points with some positions
d3.json(
  'https://gist.githubusercontent.com/will-r-chase/375d6366e6c32caf3862d1f6154f87a0/raw/f632753fc5940ac57e55276f38bca2262cb87907/landers_before2.geojson'
)
  .then(d => clean(d))
  .then(data => {
    const svgWidth = 700;
    const svgHeight = 500;
    const circleRad = 10;

    //set up a scale for when the points become a timeline
    const timeScaleTriggered = d3
      .scaleTime()
      .domain(d3.extent(data.features, d => d.properties.day))
      .range([circleRad, svgWidth - circleRad]);

    //set up SVG to fill wrapper
    const svg = d3
      .select('svg')
      .attr('preserveAspectRatio', 'xMinYMin meet')
      .attr('viewBox', `0 0 ${svgWidth} ${svgHeight}`);

    const g = svg.append('g');

    //create some circles from our data with a random position and color
    //initially they have radius of 0 so they're not visible right away
    let circles = g
      .selectAll('circle')
      .data(data.features)
      .join('circle')
      .attr('class', 'points')
      .attr('r', 0)
      .attr('cx', () => Math.random() * svgWidth)
      .attr('cy', () => Math.random() * svgHeight)
      .style('fill', () => rainbow[Math.floor(Math.random() * rainbow.length)])
      .style('opacity', 0.7);

    //sets up the class toggle on each scrolling text box
    //so that it becomes opaque when in view and transparent when exiting
    gsap.utils.toArray('.step').forEach(step => {
      ScrollTrigger.create({
        trigger: step,
        start: 'top 80%',
        end: 'center top',
        toggleClass: 'active',
        markers: true,
        id: 'toggle-active-class'
      });
    });

    //The initial animation to show the points
    //sets the point radius to a random value from 0 to 20
    gsap.to('.points', {
      scrollTrigger: {
        trigger: '#step-1',
        start: 'top center',
        toggleActions: 'play none none reverse',
        markers: true,
        id: 'first-box'
      },
      attr: {r: () => Math.random() * 20},
      duration: 0.5,
      ease: 'power3.out'
    });

    //the animation triggered by the second text box
    //shuffles the X position of the points to a random value
    gsap.to('.points', {
      scrollTrigger: {
        trigger: '#step-2',
        start: 'top center',
        toggleActions: 'play none none reverse',
        markers: true,
        id: 'second-box'
      },
      attr: {cx: () => Math.random() * svgWidth},
      duration: 0.5,
      ease: 'power3.inOut'
    });

    //the animation triggered by the third text box
    //this just sets up the scroll trigger, but the animation
    //is done using our D3 functions, passed as callbacks to onEnter and onLeaveBack
    ScrollTrigger.create({
      trigger: '#step-3',
      start: 'top center',
      onEnter: circlesToTimeline,
      onLeaveBack: circlesToRandom,
      markers: true,
      id: 'third-box'
    });

    //This pins the SVG chart wrapper when it hits the center of the viewport
    //and releases the pin when the final textbox meets the bottom of the chart
    //we use a function to define the end point to line up the bottom of the
    //text box with the bottom of the chart
    ScrollTrigger.create({
      trigger: '#chart-wrapper',
      endTrigger: '#step-4',
      start: 'center center',
      end: () => {
        const height = window.innerHeight;
        const chartHeight = document.querySelector('#chart-wrapper')
          .offsetHeight;
        return `bottom ${chartHeight + (height - chartHeight) / 2}px`;
      },
      pin: true,
      pinSpacing: false,
      markers: true,
      id: 'chart-pin'
    });

    //scrubbing animation
    //sets an animation on each stacked text element
    //but gives each one a slightly different scrub value
    //so when you scroll they separate and catch up at
    //different rates
    gsap.utils.toArray('.scrub').forEach((el, i) => {
      gsap.to(el, {
        scrollTrigger: {
          trigger: '.scrub-wrapper',
          start: 'top top',
          end: 'bottom center+=150',
          pin: '.scrub-wrapper',
          scrub: (7 - i) * 0.1,
          markers: true,
          id: 'scrub-tween'
        },
        y: '45vh'
      });
    });

    //our custom d3 functions that stack our circles
    //into a timeline dot plot
    function circlesToTimeline() {
      circles
        .transition()
        .duration(1000)
        .attr('r', circleRad)
        .attr('cx', d => timeScaleTriggered(d.properties.day))
        .attr('cy', d => svgHeight - d.properties.id_day * 20)
        .style('opacity', 1);
    }
    //reverses the circles back to a random position
    function circlesToRandom() {
      circles
        .transition()
        .attr('r', () => Math.random() * 20)
        .attr('cx', () => Math.random() * svgWidth)
        .attr('cy', () => Math.random() * svgHeight)
        .style('opacity', 0.7);
    }

    //////////////////////////////////////////////////
    // Ignore this, it's all just for the markers   //
    // menu, nothing to do with animation           //
    //////////////////////////////////////////////////
    const pinMarkers = contains('div', 'chart-pin');
    const toggleMarkers = contains('div', 'toggle-active-class');
    const box1Markers = contains('div', 'first-box');
    const box2Markers = contains('div', 'second-box');
    const box3Markers = contains('div', 'third-box');
    const scrubMarkers = contains('div', 'scrub-tween');

    const allMarkers = [
      ...pinMarkers,
      ...toggleMarkers,
      ...box1Markers,
      ...box2Markers,
      ...box3Markers,
      ...scrubMarkers
    ];
    allMarkers.forEach(el => {
      el.classList.add('hidden');
    });

    function updateMarkers(check, markers) {
      if (check.checked) {
        markers.forEach(el => {
          el.classList.add('hidden');
        });
      } else {
        markers.forEach(el => {
          el.classList.remove('hidden');
        });
      }
    }

    pinLabel.addEventListener('click', () =>
      updateMarkers(pinCheck, pinMarkers)
    );
    toggleLabel.addEventListener('click', () =>
      updateMarkers(toggleCheck, toggleMarkers)
    );
    box1Label.addEventListener('click', () =>
      updateMarkers(box1Check, box1Markers)
    );
    box2Label.addEventListener('click', () =>
      updateMarkers(box2Check, box2Markers)
    );
    box3Label.addEventListener('click', () =>
      updateMarkers(box3Check, box3Markers)
    );
    scrubLabel.addEventListener('click', () =>
      updateMarkers(scrubCheck, scrubMarkers)
    );
  });

const timeParse = d3.timeParse('%Y-%m-%d %H:%M:%S');
function clean(data) {
  for (const d of data.features) {
    const date = timeParse(d.properties.time);
    d.properties.date = date;
    d.properties.day = d3.timeDay(date);
  }
  return data;
}`

In the project code example shown below, a plane glb model can be rendered as wireframe, and metrics shown 
`// clearing the console (just a CodePen thing)

console.clear();

// there are 3 parts to this
//
// Scene:           Setups and manages threejs rendering
// loadModel:       Loads the 3d obj file
// setupAnimation:  Creates all the GSAP 
//                  animtions and scroll triggers 
//
// first we call loadModel, once complete we call
// setupAnimation which creates a new Scene

class Scene
{
	constructor(model)
	{
		this.views = [
			{ bottom: 0, height: 1 },
			{ bottom: 0, height: 0 }
		];
		
		this.renderer = new THREE.WebGLRenderer({
			antialias: true,
			alpha: true
		});
		
		this.renderer.setSize(window.innerWidth, window.innerHeight);
		this.renderer.shadowMap.enabled = true;
		this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
		this.renderer.setPixelRatio(window.devicePixelRatio);

		document.body.appendChild( this.renderer.domElement );
		
		// scene

		this.scene = new THREE.Scene();
		
		for ( var ii = 0; ii < this.views.length; ++ ii ) {

			var view = this.views[ ii ];
			var camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 1, 2000 );
			camera.position.fromArray([0, 0, 180] );
			camera.layers.disableAll();
			camera.layers.enable( ii );
			view.camera = camera;
			camera.lookAt(new THREE.Vector3(0, 5, 0));
		}
		
		//light

		this.light = new THREE.PointLight( 0xffffff, 0.75 );
		this.light.position.z = 150;
		this.light.position.x = 70;
		this.light.position.y = -20;
		this.scene.add( this.light );

		this.softLight = new THREE.AmbientLight( 0xffffff, 1.5 );
		this.scene.add(this.softLight)

		// group

		this.onResize();
		window.addEventListener( 'resize', this.onResize, false );
		
		var edges = new THREE.EdgesGeometry( model.children[ 0 ].geometry );
		let line = new THREE.LineSegments( edges );
		line.material.depthTest = false;
		line.material.opacity = 0.5;
		line.material.transparent = true;
		line.position.x = 0.5;
		line.position.z = -1;
		line.position.y = 0.2;	
		
		this.modelGroup = new THREE.Group();
		
		model.layers.set( 0 );
		line.layers.set( 1 );
			
		this.modelGroup.add(model);
		this.modelGroup.add(line);
		this.scene.add(this.modelGroup);
	}
	
	render = () =>
	{
		for ( var ii = 0; ii < this.views.length; ++ ii ) {

			var view = this.views[ ii ];
			var camera = view.camera;

			var bottom = Math.floor( this.h * view.bottom );
			var height = Math.floor( this.h * view.height );

			this.renderer.setViewport( 0, 0, this.w, this.h );
			this.renderer.setScissor( 0, bottom, this.w, height );
			this.renderer.setScissorTest( true );

			camera.aspect = this.w / this.h;
			this.renderer.render( this.scene, camera );
		}
	}
	
	onResize = () => 
	{
		this.w = window.innerWidth;
		this.h = window.innerHeight;
		
		for ( var ii = 0; ii < this.views.length; ++ ii ) {
			var view = this.views[ ii ];
			var camera = view.camera;
			camera.aspect = this.w / this.h;
			let camZ = (screen.width - (this.w * 1)) / 3;
			camera.position.z = camZ < 180 ? 180 : camZ;
			camera.updateProjectionMatrix();
		}

		this.renderer.setSize( this.w, this.h );		
		this.render();
	}
}

function loadModel() 
{
	gsap.registerPlugin(ScrollTrigger);
	gsap.registerPlugin(DrawSVGPlugin);
	gsap.set('#line-length', {drawSVG: 0})
	gsap.set('#line-wingspan', {drawSVG: 0})
	gsap.set('#circle-phalange', {drawSVG: 0})
	
	var object;

	function onModelLoaded() {
		object.traverse( function ( child ) {
			let mat = new THREE.MeshPhongMaterial( { color: 0x171511, specular: 0xD0CBC7, shininess: 5, flatShading: true } );
			child.material = mat;
		});

		setupAnimation(object);
	}

	var manager = new THREE.LoadingManager( onModelLoaded );
	manager.onProgress = ( item, loaded, total ) => console.log( item, loaded, total );

	var loader = new THREE.OBJLoader( manager );
	loader.load( 'https://assets.codepen.io/557388/1405+Plane_1.obj', function ( obj ) { object = obj; });
}

function setupAnimation(model)
{
	let scene = new Scene(model);
	let plane = scene.modelGroup;
	
	gsap.fromTo('canvas',{x: "50%", autoAlpha: 0},  {duration: 1, x: "0%", autoAlpha: 1});
	gsap.to('.loading', {autoAlpha: 0})
	gsap.to('.scroll-cta', {opacity: 1})
	gsap.set('svg', {autoAlpha: 1})
	
	let tau = Math.PI * 2;

	gsap.set(plane.rotation, {y: tau * -.25});
	gsap.set(plane.position, {x: 80, y: -32, z: -60});
	
	scene.render();
	
	var sectionDuration = 1;
	gsap.fromTo(scene.views[1], 
		{ 	height: 1, bottom: 0 }, 
		{
			height: 0, bottom: 1,
			ease: 'none',
			scrollTrigger: {
			  trigger: ".blueprint",
			  scrub: true,
			  start: "bottom bottom",
			  end: "bottom top"
			}
		})
	
	gsap.fromTo(scene.views[1], 
		{ 	height: 0, bottom: 0 }, 
		{
			height: 1, bottom: 0,
			ease: 'none',
			scrollTrigger: {
			  trigger: ".blueprint",
			  scrub: true,
			  start: "top bottom",
			  end: "top top"
			}
		})
	
	gsap.to('.ground', {
		y: "30%",
		scrollTrigger: {
		  trigger: ".ground-container",
		  scrub: true,
		  start: "top bottom",
		  end: "bottom top"
		}
	})
	
	gsap.from('.clouds', {
		y: "25%",
		scrollTrigger: {
		  trigger: ".ground-container",
		  scrub: true,
		  start: "top bottom",
		  end: "bottom top"
		}
	})
	
	gsap.to('#line-length', {
		drawSVG: 100,
		scrollTrigger: {
		  trigger: ".length",
		  scrub: true,
		  start: "top bottom",
		  end: "top top"
		}
	})
	
	gsap.to('#line-wingspan', {
		drawSVG: 100,
		scrollTrigger: {
		  trigger: ".wingspan",
		  scrub: true,
		  start: "top 25%",
		  end: "bottom 50%"
		}
	})	
	
	gsap.to('#circle-phalange', {
		drawSVG: 100,
		scrollTrigger: {
		  trigger: ".phalange",
		  scrub: true,
		  start: "top 50%",
		  end: "bottom 100%"
		}
	})
	
	gsap.to('#line-length', {
		opacity: 0,
		drawSVG: 0,
		scrollTrigger: {
		  trigger: ".length",
		  scrub: true,
		  start: "top top",
		  end: "bottom top"
		}
	})
	
	gsap.to('#line-wingspan', {
		opacity: 0,
		drawSVG: 0,
		scrollTrigger: {
		  trigger: ".wingspan",
		  scrub: true,
		  start: "top top",
		  end: "bottom top"
		}
	})	
	
	gsap.to('#circle-phalange', {
		opacity: 0,
		drawSVG: 0,
		scrollTrigger: {
		  trigger: ".phalange",
		  scrub: true,
		  start: "top top",
		  end: "bottom top"
		}
	})
	
	let tl = new gsap.timeline(
	{
		onUpdate: scene.render,
		scrollTrigger: {
		  trigger: ".content",
		  scrub: true,
		  start: "top top",
		  end: "bottom bottom"
		},
		defaults: {duration: sectionDuration, ease: 'power2.inOut'}
	});
	
	let delay = 0;
	
	tl.to('.scroll-cta', {duration: 0.25, opacity: 0}, delay)
	tl.to(plane.position, {x: -10, ease: 'power1.in'}, delay)
	
	delay += sectionDuration;
	
	tl.to(plane.rotation, {x: tau * .25, y: 0, z: -tau * 0.05, ease: 'power1.inOut'}, delay)
	tl.to(plane.position, {x: -40, y: 0, z: -60, ease: 'power1.inOut'}, delay)
	
	delay += sectionDuration;
	
	tl.to(plane.rotation, {x: tau * .25, y: 0,  z: tau * 0.05, ease: 'power3.inOut'}, delay)
	tl.to(plane.position, {x: 40, y: 0, z: -60, ease: 'power2.inOut'}, delay)
	
	delay += sectionDuration;
	
	tl.to(plane.rotation, {x: tau * .2, y: 0, z: -tau * 0.1, ease: 'power3.inOut'}, delay)
	tl.to(plane.position, {x: -40, y: 0, z: -30, ease: 'power2.inOut'}, delay)
	
	delay += sectionDuration;
	
	tl.to(plane.rotation, { x: 0, z: 0, y: tau * .25}, delay)
	tl.to(plane.position, { x: 0, y: -10, z: 50}, delay)
	
	delay += sectionDuration;
	delay += sectionDuration;
	
	tl.to(plane.rotation, {x: tau * 0.25, y: tau *.5, z: 0, ease:'power4.inOut'}, delay)
	tl.to(plane.position, {z: 30, ease:'power4.inOut'}, delay)
	
	delay += sectionDuration;
	
	tl.to(plane.rotation, {x: tau * 0.25, y: tau *.5, z: 0, ease:'power4.inOut'}, delay)
	tl.to(plane.position, {z: 60, x: 30, ease:'power4.inOut'}, delay)
	
	delay += sectionDuration;
	
	tl.to(plane.rotation, {x: tau * 0.35, y: tau *.75, z: tau * 0.6, ease:'power4.inOut'}, delay)
	tl.to(plane.position, {z: 100, x: 20, y: 0, ease:'power4.inOut'}, delay)
	
	delay += sectionDuration;
	
	tl.to(plane.rotation, {x: tau * 0.15, y: tau *.85, z: -tau * 0, ease: 'power1.in'}, delay)
	tl.to(plane.position, {z: -150, x: 0, y: 0, ease: 'power1.inOut'}, delay)
	
	delay += sectionDuration;
	
	tl.to(plane.rotation, {duration: sectionDuration, x: -tau * 0.05, y: tau, z: -tau * 0.1, ease: 'none'}, delay)
	tl.to(plane.position, {duration: sectionDuration, x: 0, y: 30, z: 320, ease: 'power1.in'}, delay)
	
	tl.to(scene.light.position, {duration: sectionDuration, x: 0, y: 0, z: 0}, delay)
}

loadModel();`







Here is the code for glb explosion 
`export const Drill = () => {
  const gltf = useGLTF("drill-corrected.glb");

  const mainObject = gltf.scene.getObjectByName("Main_") as Mesh;
  const targetMap: Map<string, Vector3> = new Map<string, Vector3>();
  const explosionCenter = new Vector3(0, 0.15, 0); // Higher epicenter for the drill
  const explosionFactor = 0.8; // Move uniformly along the direction for every part.

  useEffect(() => {
    gltf.scene.traverse((object) => {
      if (object.uuid !== mainObject.uuid) { // Discard the main object so it is frozen.
        const vector = object.position.clone().sub(explosionCenter).normalize();

        const displacement = object.position
          .clone()
          .add(
            vector.multiplyScalar(
              object.position.distanceTo(explosionCenter) * explosionFactor
            )
          );
        targetMap.set(object.name, displacement); // Store it in a dictionnary for later use.
      }
    });
  }, []);


  return (
    <>
      <primitive object={gltf.scene}></primitive>
    </>
  );
};

gltf.scene.traverse((object) => {
        if (object.uuid !== mainObject.uuid) {
          const displacement = targetMap.get(object.name) as Vector3;
          const tl = gsap.timeline().to(object.position, {
            x: displacement.x,
            y: displacement.y,
            z: displacement.z,
            duration: 5,
          });
        }
      });
      
      const [trigger, setTrigger] = useState(false);

  useEffect(() => {
    gltf.scene.traverse((object) => {
      if (object.uuid !== mainObject.uuid) {
        const vector = object.position.clone().sub(explosionCenter).normalize();

        const displacement = object.position
          .clone()
          .add(
            vector.multiplyScalar(
              object.position.distanceTo(explosionCenter) * explosionFactor
            )
          );
        targetMap.set(object.name, displacement);
      }
    });
    setTrigger(true);
  }, []);

  useEffect(() => {
    if (!trigger) {
      gltf.scene.traverse((object) => {
        if (object.uuid !== mainObject.uuid) {
          const displacement = targetMap.get(object.name) as Vector3;
          const tl = gsap.timeline().to(object.position, {
            x: displacement.x,
            y: displacement.y,
            z: displacement.z,
            duration: 5,
          });
        }
      });
    }
  }, [trigger]);
  
  let scrollTriggerParams = {
        toggleActions: "play none none reverse",
        scrub: 1.5,
      };
      gltf.scene.traverse((object) => {
        if (object.uuid !== mainObject.uuid) {
          const displacement = targetMap.get(object.name) as Vector3;
          gsap
            .timeline({
              scrollTrigger: {
                ...scrollTriggerParams,
              },
            })
            .to(object.position, {
              x: displacement.x,
              y: displacement.y,
              z: displacement.z,
            });
        }
      });`