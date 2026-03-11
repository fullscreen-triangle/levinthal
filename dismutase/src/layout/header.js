import React from 'react'

export default function Header({handleOnClick, ActiveIndex}) {

    return (
        <>
            {/* HEADER */}
            <div className="cavani_tm_header">
                <div className="logo">
                    <a href="#" style={{textDecoration: 'none', color: '#fff', fontFamily: 'Poppins', fontWeight: 700, fontSize: '18px'}}>DISMUTASE</a>
                </div>
                <div className="menu">
                    <ul className="transition_link">
                        <li onClick={() => handleOnClick(0)}><a className={ActiveIndex === 0 ? "active" : ""}>Home</a></li>
                        <li onClick={() => handleOnClick(1)}><a className={ActiveIndex === 1 ? "active" : ""}>Dynamics</a></li>
                        <li onClick={() => handleOnClick(2)}><a className={ActiveIndex === 2 ? "active" : ""}>Folding</a></li>
                        <li onClick={() => handleOnClick(3)}><a className={ActiveIndex === 3 ? "active" : ""}>Catalysis</a></li>
                        <li onClick={() => handleOnClick(4)}><a className={ActiveIndex === 4 ? "active" : ""}>Publications</a></li>
                        <li onClick={() => handleOnClick(5)}><a className={ActiveIndex === 5 ? "active" : ""}>Contact</a></li>
                    </ul>
                </div>
            </div>
            {/* /HEADER */}

        </>
    )
}
