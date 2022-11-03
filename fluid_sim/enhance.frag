#version 330 core
in vec3 ourColor;
in vec2 ourTexCoord;
uniform sampler2D ourTexture1;
uniform sampler2D ourTexture2;
out vec4 FragColor;

void main()
{
	int r = 5;
    vec2 texSize = vec2(380, 380);
	vec4 color = vec4(0,0,0,0);
    for (int y = -r; y <= r; y++) {
        for (int x = -r; x <= r; x++) {
				color += texture(ourTexture2, ourTexCoord + (vec2(x, y)/texSize));
        }
    }
    vec4 blurColor = color / ((2 * r + 1) * (2 * r + 1));
    vec4 realColor =  texture(ourTexture2, ourTexCoord);
    FragColor = (realColor - blurColor)*1.5+realColor;
}