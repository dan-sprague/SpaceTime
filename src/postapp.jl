# ---------------------------------------------------------------------------
# Standalone post-processing app: loads a raw HDR frame (save_raw) and
# grades it interactively. Everything here is pure image-space — anything
# that needs the rays (framing, DoF, motion blur, disc shading, volumetrics)
# lives in the tracers.
# ---------------------------------------------------------------------------

"""
    postprocessor(rawpath; title=basename(rawpath))

Interactive grading app for a raw frame saved by [`save_raw`](@ref):
exposure/gamma/contrast, tonemap (ACES ↔ Reinhard, hue preserve), bloom,
star streaks, sensor model (ISO/read noise), vignette, lens distortion and
Auto balance — with a live preview on a ≤960-wide proxy and an
"Export PNG" button that runs the full-resolution pipeline and writes
`<raw>_graded.png`.

Launch with `julia -t auto,1 --project post_demo.jl <raw.tiff>` so grading
runs on a worker thread. Returns the `Figure`.
"""
function postprocessor(rawpath::AbstractString; title::String=basename(rawpath))
    img_raw, meta = load_raw(rawpath)
    if haskey(meta, "camera_pos")
        p = meta["camera_pos"]
        title *= string("  ·  traced from r = ",
                        round(sqrt(sum(abs2, p)); digits=2), "M")
    end
    w, h = size(img_raw)
    # Live grading runs on a subsampled proxy; export uses the full frame.
    stride = max(1, cld(w, 960))
    proxy = img_raw[1:stride:end, 1:stride:end]

    fig = Figure(size=(1360, 860))
    ax = GLMakie.Axis(fig[1, 1], aspect=DataAspect(), title=title)
    img_obs = Observable(map(clamp01nan, proxy))
    image!(ax, img_obs)
    hidedecorations!(ax)
    for k in (:rectanglezoom, :dragpan, :scrollzoom, :limitreset)
        deregister_interaction!(ax, k)
    end
    colsize!(fig.layout, 1, GLMakie.Relative(0.68))

    ctrl = GridLayout(fig[1, 2]; valign=:top, tellheight=false)
    crow = 1
    Label(ctrl[crow, 1], "Post-processing"; fontsize=16, halign=:left)
    crow += 1
    sg = SliderGrid(
        ctrl[crow, 1],
        (label = "Gain", range = 0.0:0.01:2.0, format = "{:.2f}", startvalue = 1.0),
        (label = "Exposure (EV)", range = -5.0:0.1:5.0, format = "{:.1f}", startvalue = 0.0),
        (label = "Gamma", range = 0.1:0.05:3.0, format = "{:.2f}", startvalue = 2.2),
        (label = "Bloom strength", range = 0.0:0.05:2.0, format = "{:.2f}", startvalue = 0.6),
        (label = "Bloom threshold", range = 0.0:0.05:2.0, format = "{:.2f}", startvalue = 0.5),
        (label = "Bloom radius", range = 1.0:1.0:50.0, format = "{:.0f}", startvalue = 15.0),
        (label = "Bloom power", range = 0.1:0.1:3.0, format = "{:.1f}", startvalue = 1.5),
        (label = "Streak strength", range = 0.0:0.05:2.0, format = "{:.2f}", startvalue = 0.0),
        (label = "Streak length", range = 0.05:0.05:1.0, format = "{:.2f}", startvalue = 0.4),
        (label = "Streak width", range = 0.5:0.5:5.0, format = "{:.1f}", startvalue = 1.5),
        (label = "Star spikes", range = 2:1:8, format = "{:.0f}", startvalue = 4),
        (label = "Color preserve", range = 0.0:0.05:1.0, format = "{:.2f}", startvalue = 0.75),
        (label = "Contrast", range = -1.0:0.05:1.0, format = "{:.2f}", startvalue = 0.0),
        (label = "ISO", range = 50.0:50.0:12800.0, format = "{:.0f}", startvalue = 100.0),
        (label = "Read noise (e⁻)", range = 0.0:0.1:10.0, format = "{:.1f}", startvalue = 0.0),
        (label = "Vignette", range = 0.0:0.05:1.0, format = "{:.2f}", startvalue = 0.0),
        (label = "Distortion k1", range = -0.2:0.005:0.2, format = "{:.3f}", startvalue = 0.0),
        tellwidth = false, tellheight = true
    )
    crow += 1
    tgrid = GridLayout(ctrl[crow, 1])
    Label(tgrid[1, 1], "ACES tonemap"; halign=:left)
    aces_toggle = Toggle(tgrid[1, 2]; active=true)
    Label(tgrid[2, 1], "Auto balance"; halign=:left)
    ab_toggle = Toggle(tgrid[2, 2]; active=false)
    crow += 1
    export_btn = Button(ctrl[crow, 1]; label="Export PNG", halign=:left)
    crow += 1
    status_obs = Observable("Ready — grading $(w)×$(h) via a $(size(proxy, 1))×$(size(proxy, 2)) proxy")
    Label(ctrl[crow, 1], status_obs; halign=:left, tellwidth=false)
    rowgap!(ctrl, 8)

    # Parameter snapshot, written only on thread 1 (slider callbacks), read
    # by the grading worker.
    grab() = (; gain=sg.sliders[1].value[], exposure=sg.sliders[2].value[],
              gamma=sg.sliders[3].value[], bloom=sg.sliders[4].value[],
              threshold=sg.sliders[5].value[], radius=sg.sliders[6].value[],
              power=sg.sliders[7].value[], sstr=sg.sliders[8].value[],
              slen=sg.sliders[9].value[], swid=sg.sliders[10].value[],
              spikes=Int(round(sg.sliders[11].value[])),
              preserve=sg.sliders[12].value[], contrast=sg.sliders[13].value[],
              iso=sg.sliders[14].value[], read_noise=sg.sliders[15].value[],
              vignette=sg.sliders[16].value[], k1=sg.sliders[17].value[],
              aces=aces_toggle.active[], autobalance=ab_toggle.active[])
    params_ref = Ref(grab())

    function apply_post(src, p)
        img = postprocess(src; gain=p.gain, exposure=p.exposure, gamma=p.gamma,
                          bloom_strength=p.bloom, threshold=p.threshold,
                          bloom_radius=p.radius, bloom_power=p.power,
                          streak_strength=p.sstr, streak_length=p.slen,
                          streak_width=p.swid, n_spikes=p.spikes,
                          tonemap=p.aces ? :aces : :reinhard,
                          tonemap_hue_preserve=p.preserve, contrast=p.contrast)
        if p.iso != 100.0 || p.read_noise > 0.0
            sensor_expose!(img; iso=p.iso, t_exp=1.0, read_noise_e=p.read_noise,
                           saturation=1.0e6)
        end
        p.vignette > 0.0 && apply_vignette!(img; strength=p.vignette)
        p.k1 != 0.0 && apply_lens_distortion!(img; k1=p.k1)
        p.autobalance && auto_balance!(img)
        return map(clamp01nan, img)
    end

    # Latest-wins grading worker, same pattern as the render previews.
    version = Threads.Atomic{Int}(0)
    wakeup = Channel{Nothing}(1)
    out_chan = Channel{Tuple{Matrix{RGBf},Float64}}(1)
    function request_grade()
        params_ref[] = grab()
        Threads.atomic_add!(version, 1)
        isready(wakeup) || put!(wakeup, nothing)
        return nothing
    end
    Threads.@spawn begin
        done = 0
        try
            while true
                take!(wakeup)
                while done < version[]
                    v = version[]
                    t0 = time()
                    img = try
                        apply_post(proxy, params_ref[])
                    catch e
                        e isa InvalidStateException && rethrow()
                        @error "Grade failed" exception=(e, catch_backtrace())
                        nothing
                    end
                    done = v
                    if img !== nothing
                        while isready(out_chan)
                            take!(out_chan)
                        end
                        put!(out_chan, (img, time() - t0))
                    end
                end
            end
        catch e
            e isa InvalidStateException || rethrow()
        end
    end
    @async try
        for (img, elapsed) in out_chan
            img_obs[] = img
            status_obs[] = string("Grade: ", round(1000 * elapsed; digits=0), " ms")
        end
    catch e
        e isa InvalidStateException || rethrow()
    end
    on(events(fig.scene).window_open) do open
        if !open
            close(wakeup)
            close(out_chan)
        end
    end

    for sl in sg.sliders
        on(_ -> request_grade(), sl.value)
    end
    on(_ -> request_grade(), aces_toggle.active)
    on(_ -> request_grade(), ab_toggle.active)

    exporting = Ref(false)
    on(export_btn.clicks) do _
        exporting[] && return
        exporting[] = true
        p = grab()
        export_btn.label[] = "Exporting…"
        result = Channel{Any}(1)
        Threads.@spawn begin
            try
                img = apply_post(img_raw, p)
                out = splitext(rawpath)[1] * "_graded.png"
                FileIO.save(out, rotr90(img))
                put!(result, (:ok, out))
            catch e
                @error "Export failed" exception=(e, catch_backtrace())
                put!(result, (:error, e))
            end
        end
        @async begin
            res = take!(result)
            exporting[] = false
            export_btn.label[] = "Export PNG"
            status_obs[] = res[1] === :ok ? "Exported: $(res[2])" :
                           "Export error: $(res[2])"
        end
    end

    display(fig)
    request_grade()
    return fig
end
