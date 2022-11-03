#version 330 core
in vec2 texCoord;
out vec4 FragColor;
uniform sampler2D ourTexture;
uniform vec2  texImgSize;

void main()
{	        
    //FragColor = texture(ourTexture, texCoord);
    FragColor = texture(ourTexture, texCoord-vec2(1,0)/texImgSize);
}