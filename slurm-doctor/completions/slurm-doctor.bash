# bash completion for slurm-doctor
_slurm_doctor() {
    local cur prev cmds
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    cmds="collect suggest patch heal sweep rules"

    # global flags
    local gflags="--version --verbose -v --cache-dir --report-dir --llm"

    # find the subcommand, if any
    local i sub=""
    for ((i = 1; i < COMP_CWORD; i++)); do
        case "${COMP_WORDS[i]}" in
            collect|suggest|patch|heal|sweep|rules) sub="${COMP_WORDS[i]}"; break ;;
        esac
    done

    if [[ -z "$sub" ]]; then
        COMPREPLY=( $(compgen -W "$cmds $gflags" -- "$cur") )
        return 0
    fi

    case "$sub" in
        suggest)  COMPREPLY=( $(compgen -W "--refresh --from-hook" -- "$cur") ) ;;
        collect)  COMPREPLY=( $(compgen -W "--refresh" -- "$cur") ) ;;
        patch)    COMPREPLY=( $(compgen -W "--refresh" -- "$cur") ) ;;
        heal)     COMPREPLY=( $(compgen -W "--refresh --yes" -- "$cur") ) ;;
        sweep)    COMPREPLY=( $(compgen -W "--since --refresh --states" -- "$cur") ) ;;
    esac

    # offer recently-failed job ids for the jobid slot
    if [[ "$sub" =~ ^(collect|suggest|patch|heal)$ && "$cur" != -* ]]; then
        local ids
        ids=$(sacct -X --noheader --parsable2 --format=JobIDRaw \
              --state=F,TO,OOM,NF,BF,DL --starttime=now-1day 2>/dev/null | head -50)
        COMPREPLY+=( $(compgen -W "$ids" -- "$cur") )
    fi
    return 0
}
complete -F _slurm_doctor slurm-doctor
