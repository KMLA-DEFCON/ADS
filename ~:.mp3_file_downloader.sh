function youtube_magic() {
    read "filetype?Enter the type of file: "
    read "Multi?Multiple URLS?(Y/N): "

    if [ "$Multi" = "Y" ]
    then
        echo "Write one URL per line. Finish with ctrl+D."
        if [ "$filetype" = "mp3" ]
        then
            youtube-dl -x --audio-format mp3 --audio-quality 0 -a - -o '~/Music/youtube-dl/%(title)s.%(ext)s'
        elif [ "$filetype" = "mp4" ]
        then
            youtube-dl --format mp4 -f best -a - -o '~/Movies/youtube-dl/%(title)s.%(ext)s'
        fi
    elif [ "$Multi" = "N" ]
    then
        read "URL?Enter the URL of video: "
        if [ "$filetype" = "mp3" ]
        then
            youtube-dl -x --audio-format mp3 --audio-quality 0 $URL -o '~/Music/youtube-dl/%(title)s.%(ext)s'
        elif [ "$filetype" = "mp4" ]
        then
            youtube-dl --format mp4 -f best $URL -o '~/Movies/youtube-dl/%(title)s.%(ext)s'
        fi
    else
        echo "Something Broke!"
    fi
}
