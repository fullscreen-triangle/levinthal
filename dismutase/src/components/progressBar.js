const ProgressBar = (props) => {
  const { bgcolor, completed, title } = props;

  const containerStyles = {
    height: 3,
    width: '100%',
    minWidth: '100%',
    backgroundColor: "#222",
    borderRadius: 50,
    marginBottom:55,

  }

  const fillerStyles = {
    height: '100%',
    width: `${completed}%`,
    backgroundColor: bgcolor,
    borderRadius: 'inherit'
  }

  const labelWraper = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'start',
    position:'relative',
    top:'-35px'
  }

  const labelStyles = {
    paddingTop:5,
    paddingBottom:5,
    color: '#ccc'
  }

  return (
    <div style={containerStyles}>
      <div style={fillerStyles}>
        <div style={labelWraper}>
          <span style={labelStyles}>{`${title}`}</span>
          <span style={labelStyles}>{`${completed}%`}</span>
        </div>
      </div>
    </div>
  );
};



export default ProgressBar;