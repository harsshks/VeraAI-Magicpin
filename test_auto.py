from bot import conversations, reply, ReplyBody
import asyncio

async def test():
    for i in range(3):
        res = await reply(ReplyBody(
            conversation_id='test1', merchant_id='m1', from_role='merchant',
            message='Thank you for contacting Dr. Meera\'s Dental Clinic! Our team will respond shortly.',
            received_at='now', turn_number=i+1))
        state = conversations.get('test1')
        print(f"Turn {i+1}: count={state.auto_reply_count}, action={res.get('action')}")

asyncio.run(test())
