set_project("cuda-lab")

add_rules("mode.debug", "mode.release")

target("radix_sort")
    set_languages("cxx20")
    set_kind("binary")
    set_warnings("all", "extra", "pedantic")

    add_files("src/radix_sort.cu")
    add_cugencodes("native")
    add_cugencodes("compute_120")

    after_build(function (target)
        os.cp(target:targetfile(), "bin/")
    end)
target_end()
