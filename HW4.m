clc
clear
close all

D=1.49597870e8;        %km  (earth sun distance)
G=6.674e-20;                % km^3/kg/s^2
m_earth= 5.9722e24;     %kg
m_sun=1.98847e30;       %kg
mu=m_earth/(m_earth+m_sun);
Gm2= G*m_earth;   
Gm1=G*m_sun;
w=2*pi/365.25/86400;        % rad/s
tspan=linspace(0,10*365.25*86400,2000);
x_I=-mu*D;                  % km
x_II=(1-mu)*D;
y_I=0;
y_II=0;
z_I=0;
z_II=0;
r2=2000000;         %km
r1=D+r2;    %km
y0=[x_II+r2 2 0 5 5000 5];
% y0=[x_II+r2 2 0 5 5000000 5];
[t,xyz]=ode45(@(t,y) EOM(t,y,x_I,x_II,y_I,y_II,z_I,z_II,r2,r1,w,Gm2,Gm1),tspan,y0);

x=xyz(:,1);
y=xyz(:,3);
z=xyz(:,5);
X=x.*cos(w*t)-y.*sin(w*t);
Y=x.*sin(w*t)+y.*cos(w*t);
Z=z;

% Earth-fixed movie
figure(1)
grid on
title('JWST in Earth-fixed frame')
npanels=20;
erad = 6378.1e4; % equatorial radius (km)
prad = 6356.8e4; % polar radius (km)
axis(2e8*[-1 1 -1 1 -1 1]);
view(9,26);
hold on;
axis vis3d;
[ xx, yy, zz ] = ellipsoid(0, 0, 0, erad, erad, prad, npanels);
globe = surf(xx, yy, -zz, 'FaceColor', '#4DBEEE', 'EdgeColor', 0.1*[1 1 1]);


curve1 = animatedline('LineWidth',0.5,'Color',[0.4940 0.1840 0.5560]);

set(gca,'XLim',[-2e8 2e8],'YLim',[-2e8 2e8],'ZLim',[-1e8 1e8]);
view(43,24);
hold on;
for i=1:length(tspan)
    addpoints(curve1,x(i),y(i),z(i));
    head = scatter3(x(i),y(i),z(i),'filled','MarkerFaceColor','k');
    drawnow
    pause(0.0001);
    if i~=length(tspan)
        delete(head);
    end
end

legend('Earth','JWST')
xlabel('X (km)')
ylabel('Y (km)')
zlabel('Z (km)')

% Solar system movie
re=149600000*0.8; %km
Xe=re*cos(w*t);
Ye=re*sin(w*t);
Ze=zeros(length(tspan));

figure(2)
title('JWST in solar system frame')
npanels=20;
erad = 6955080; % equatorial radius (km)
prad = 6955080; % polar radius (km)
axis(2e8*[-1 1 -1 1 -1 1]);
view(9,26);
hold on;
axis vis3d;
[ xx, yy, zz ] = ellipsoid(0, 0, 0, erad, erad, prad, npanels);
globe = surf(xx, yy, -zz, 'FaceColor', [0.8500 0.3250 0.0980],'EdgeColor',[1 1 0]);
grid on

curve1 = animatedline('LineWidth',0.5,'Color',[0.4940 0.1840 0.5560]);
curve2 = animatedline('LineWidth',3,'Color','k');
set(gca,'XLim',[-2e8 2e8],'YLim',[-2e8 2e8],'ZLim',[-1e8 1e8]);
view(43,24);
hold on;
for i=1:length(tspan)
    addpoints(curve1,X(i),Y(i),Z(i));
    addpoints(curve2,Xe(i),Ye(i),Ze(i));
    head = scatter3(Xe(i),Ye(i),Ze(i),'filled','MarkerFaceColor','b');
    drawnow
    pause(0.001);
    if i~=length(tspan)
        delete(head);
    end
end
hold on
legend('Sun','JWST','Earth')

xlabel('X (km)')
ylabel('Y (km)')
zlabel('Z (km)')



% Functions
function xyz=EOM(t,y,x_I,x_II,y_I,y_II,z_I,z_II,r2,r1,w,Gm2,Gm1)
    xyz=zeros(6,1);
    xyz(1)=y(2);
    xyz(2)=Gm1/r1^3*(x_I-y(1))+Gm2/r2^3*(x_II-y(1))+2*y(4)*w+y(1)*w^2;
    xyz(3)=y(4);
    xyz(4)=Gm1/r1^3*(y_I-y(3))+Gm2/r2^3*(y_II-y(3))-2*y(2)*w+y(3)*w^2;
    xyz(5)=y(6);
    xyz(6)=Gm1/r1^3*(z_I-y(5))+Gm2/r2^3*(z_II-y(5));

end


