#include <RTLib/Core/Transform.h>
#include <RTLib/Scene/Scene.h>
#include <RTLib/Scene/Object.h>
#include <RTLib/Scene/Transform.h>
#include <RTLib/Scene/Camera.h>
#include <RTLib/Scene/AnimateTransform.h>
int main()
{
	auto scene   = RTLib::Scene::Scene();
	auto object0 = RTLib::Scene::Object::New("Object_0", RTLib::Core::Transform::from_translation(RTLib::Vector3(+1.0f, 0.0f, 0.0f)));
	auto object1 = RTLib::Scene::Object::New("Object_1", RTLib::Core::Transform::from_translation(RTLib::Vector3(-1.0f, 0.0f, 0.0f)));

	scene.attach_child(object0->get_transform());
	scene.attach_child(object1->get_transform());

	auto camera  = object0->add_component<RTLib::Scene::Camera>();
	camera->set_type(RTLib::Core::CameraType::ePerspective);
	camera->set_field_of_View(30.0f);
	camera->set_aspect(1.0f);
	camera->set_near_clip_plane(0.01f);
	camera->set_far_clip_plane(100.f);

	auto animateTransform = object1->add_component<RTLib::Scene::AnimateTransform>();
	animateTransform->add_local_position(0.01f, RTLib::Vector3(0.0f));
	animateTransform->add_local_scaling (0.01f, RTLib::Vector3(3.0f));
	animateTransform->add_local_rotation(0.01f, RTLib::Quat(1.0f,0.0f,0.0f,0.0f));

	return 0;
}