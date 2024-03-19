#! /bin/env zsh

# List of font-weights
font_weights=(
    "Bold"
    "BoldItalic"
    "ExtraBold"
    "ExtraBoldItalic"
    "ExtraLight"
    "ExtraLightItalic"
    "Italic"
    "Light"
    "LightItalic"
    "Medium"
    "MediumItalic"
    "Regular"
    "SemiBold"
    "SemiBoldItalic"
    "SemiWideBold"
    "SemiWideBoldItalic"
    "SemiWideExtraBold"
    "SemiWideExtraBoldItalic"
    "SemiWideExtraLight"
    "SemiWideExtraLightItalic"
    "SemiWideItalic"
    "SemiWideLight"
    "SemiWideLightItalic"
    "SemiWideMedium"
    "SemiWideMediumItalic"
    "SemiWideRegular"
    "SemiWideSemiBold"
    "SemiWideSemiBoldItalic"
    "WideBold"
    "WideBoldItalic"
    "WideExtraBold"
    "WideExtraBoldItalic"
    "WideExtraLight"
    "WideExtraLightItalic"
    "WideItalic"
    "WideLight"
    "WideLightItalic"
    "WideMedium"
    "WideMediumItalic"
    "WideRegular"
    "WideSemiBold"
    "WideSemiBoldItalic"
)
monaspace_fonts=(
    "MonaspaceArgon"
    "MonaspaceNeon"
    "MonaspaceXenon"
    "MonaspaceKrypton"
)

# Clone from github
[[ -d monaspace ]] && rm -rf monaspace
echo "Cloning monaspace from github"
git clone https://github.com/githubnext/monaspace.git --quiet
echo "Installing monaspace fonts"

# Move to otf folder
cd monaspace/fonts/otf

# Remove the fonts if they already exist
for font in $monaspace_fonts; do
    for weight in $font_weights; do
        [[ -d /usr/share/fonts/$font-$weight.otf ]] \
        && sudo rm /usr/share/fonts/$font-$weight.otf
    done
    [[ -d /usr/share/fonts/${font}VarVF[wght,wdth,slnt].ttf ]] \
    && sudo rm /usr/share/fonts/${font}VarVF[wght,wdth,slnt].ttf
done



# Install fonts
for font in $monaspace_fonts; do
    for weight in $font_weights; do
        sudo cp $font-$weight.otf /usr/share/fonts/
    done
done

# Install variable fonts
cd ../variable
for font in $monaspace_fonts; do
    temp_font=${font}VarVF[wght,wdth,slnt].ttf
    sudo cp $temp_font /usr/share/fonts/
done

# Update font cache
echo "Updating font cache"
sudo fc-cache -f -v --quiet

# Back up
cd ../../

# Delete monaspace git folder
rm -rf ./monaspace

echo "List of installed fonts:"
fc-list | grep "Monaspace"
