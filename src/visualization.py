import numpy as np
import matplotlib.pylab as plt
import plotly.graph_objects as go
import plotly.express as px


def interactive_mri(transposed_data):
    """
    Interactive MRI plot using Plotly
    Args:
        transposed nibabel loaded matrix
    """
    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    r, c = transposed_data[0].shape
    nb_frames = transposed_data.shape[0]

    fig = go.Figure(
        frames=[
            go.Frame(
                data=go.Surface(
                    z=(6.7 - frames_idx * 0.1) * np.ones((r, c)),
                    surfacecolor=np.flipud(transposed_data[(nb_frames-1) - frames_idx])
                ),
            name=str(frames_idx) # you need to name the frame for the animation to behave properly 
            ) for frames_idx in range(nb_frames)]
    )

    # Add data to be displayed before animation starts
    fig.add_trace(
        go.Surface(
            z= 6.7 * np.ones((r, c)),
            surfacecolor=np.flipud(transposed_data[(nb_frames-1)]),
            colorscale='Gray',
    #         colorscale=px.colors.sequential.Viridis,
        )
    )
    sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]

    # Layout
    fig.update_layout(
         title='Slices in volumetric data',
         width=600,
         height=600,
         scene=dict(
                    zaxis=dict(range=[-0.1, 6.8], autorange=False),
                    aspectratio=dict(x=1, y=1, z=1),
                    ),
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
         ],
         sliders=sliders
    )
    fig.show()